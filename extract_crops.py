# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
VIDEO_PATH    = "video/8hcam4.mkv"   # đường dẫn đến file video
CAM_ID        = 4                   # ID camera (số nguyên)
OUTPUT_DIR    = "crops"             # thư mục gốc lưu kết quả
RESET_CAM     = False               # True  = xóa sạch crops/cam{C}/ rồi chạy lại từ đầu
                                    # False = đánh số track NỐI TIẾP dữ liệu cũ (tích lũy)
SAVE_EVERY    = 8                   # lưu 1 crop mỗi N frame
MIN_HEIGHT    = 80                  # chiều cao upbody tối thiểu (pixel)
MAX_PER_TRACK = 40                  # số ảnh tối đa mỗi tracklet (video dài: rải nhiều hơn)
MIN_GAP       = 50                  # cách nhau tối thiểu N frame giữa 2 ảnh CÙNG track
                                    # (~2s @25fps — chống trùng + rải mẫu theo thời gian)
ZONE_POINTS   = []                  # polygon [[x,y],...] tọa độ pixel gốc — [] = không lọc
# =================================================================

import csv
import shutil
import sys
from pathlib import Path

# Ép stdout UTF-8 để in tiếng Việt không crash trên terminal cp1252 (Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2
import numpy as np
import supervision as sv

from head_detect import detect_heads
from scale_box import scale_box_down


def _real_frame_count(cap, reported: int) -> int:
    """Số frame THẬT của video (đưa con trỏ về 0 sau khi xong).

    Một số file .mkv báo FRAME_COUNT sai — phóng đại nhiều lần. Nếu frame gần
    cuối (theo metadata) KHÔNG đọc được → binary-search tìm frame cuối đọc được
    thật, để thanh tiến trình hiển thị đúng. (Vòng crop vẫn đọc tới hết video
    bất kể số này, nên sai số chỉ ảnh hưởng hiển thị.)
    """
    if reported <= 0:
        return 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, reported - 2))
    ok, _ = cap.read()
    if ok:                                    # count đáng tin (vd .mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return reported
    lo, hi = 0, reported                      # count phóng đại → tìm cuối thật
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, _ = cap.read()
        if ok: lo = mid
        else:  hi = mid - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return lo + 1


def main():
    output_dir = Path(OUTPUT_DIR)
    cam_dir = output_dir / f"cam{CAM_ID}"                 # cấu trúc sạch: crops/cam{C}/
    csv_path = output_dir / f"meta_cam{CAM_ID}.csv"

    # --- Tùy chọn: xóa sạch camera này để làm lại từ đầu ---
    if RESET_CAM and cam_dir.exists():
        shutil.rmtree(str(cam_dir))
        if csv_path.exists():
            csv_path.unlink()
        print(f"[RESET] Đã xóa dữ liệu cũ của cam{CAM_ID}")

    cam_dir.mkdir(parents=True, exist_ok=True)

    # --- Đánh số track NỐI TIẾP: tìm số track lớn nhất đang có ---
    # Nhờ vậy chạy nhiều video cho cùng 1 cam sẽ tích lũy, không đè lên nhau,
    # mà cấu trúc thư mục vẫn gọn (crops/cam{C}/track{NNNN}/).
    existing_max = 0
    for d in cam_dir.glob("track*"):
        try:
            existing_max = max(existing_max, int(d.name.replace("track", "")))
        except ValueError:
            continue
    if existing_max:
        print(f"Cam{CAM_ID} đã có track tới {existing_max:04d} — "
              f"đánh số tiếp từ track{existing_max + 1:04d}")

    # Ánh xạ ID của ByteTrack (reset về 1 mỗi lần chạy) → số track toàn cục
    id_map: dict[int, int] = {}

    def global_track_id(bt_id: int) -> int:
        """Cấp số track nối tiếp cho mỗi ID ByteTrack mới gặp (khi nó lưu ảnh)."""
        if bt_id not in id_map:
            id_map[bt_id] = existing_max + len(id_map) + 1
        return id_map[bt_id]

    # --- Mở video ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {VIDEO_PATH}")
        sys.exit(1)

    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = _real_frame_count(cap, reported_frames)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    note = "" if total_frames == reported_frames else f"  (metadata báo {reported_frames}, đã chỉnh)"
    print(f"Video     : {VIDEO_PATH}")
    print(f"Frames    : {total_frames}{note}  |  FPS: {fps:.1f}  |  Camera ID: {CAM_ID}")
    print(f"Lưu mỗi   : {SAVE_EVERY} frame  |  Min height: {MIN_HEIGHT}px  |  "
          f"Max/track: {MAX_PER_TRACK}  |  Min gap: {MIN_GAP}")

    # Khởi tạo ByteTrack (supervision)
    tracker = sv.ByteTrack()

    track_counts: dict[int, int] = {}      # bt_id → số ảnh đã lưu
    track_last_saved: dict[int, int] = {}  # bt_id → frame của ảnh lưu gần nhất
    meta_rows: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- Detect đầu người (chạy mỗi frame để tracker hoạt động ổn định) ---
        heads = detect_heads(frame)

        if heads:
            xyxy = np.array(
                [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in heads], dtype=np.float32
            )
            confs = np.array([c for *_, c in heads], dtype=np.float32)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confs,
                class_id=np.zeros(len(heads), dtype=int),
            )
        else:
            detections = sv.Detections.empty()

        # --- Cập nhật tracker mỗi frame để giữ ID liên tục ---
        tracked = tracker.update_with_detections(detections)

        # Chỉ lưu ảnh khi đến frame được chọn
        if frame_idx % SAVE_EVERY != 0:
            if frame_idx % 500 == 0:
                print(f"  frame {frame_idx}/{total_frames} — "
                      f"{len(track_counts)} track, {len(meta_rows)} ảnh")
            continue

        if tracked is None or len(tracked) == 0 or tracked.tracker_id is None:
            continue

        for i in range(len(tracked)):
            bt_id = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)

            # Mở rộng box đầu → vùng upbody
            ub_x1, ub_y1, ub_x2, ub_y2 = scale_box_down(
                (x1, y1, x2, y2), frame.shape
            )

            # Lọc upbody quá nhỏ
            if (ub_y2 - ub_y1) < MIN_HEIGHT:
                continue

            # Lọc theo zone (nếu có): kiểm tra bottom-center (chân người)
            if ZONE_POINTS:
                cx = (ub_x1 + ub_x2) // 2
                cy = ub_y2  # dùng chân người để khớp với zone vẽ trên sàn
                zone_arr = np.array(ZONE_POINTS, dtype=np.int32)
                if cv2.pointPolygonTest(zone_arr, (float(cx), float(cy)), False) < 0:
                    continue  # ngoài zone → bỏ qua

            # Loại crop degenerate (sau clamp vẫn bị rỗng)
            if ub_x2 <= ub_x1 or ub_y2 <= ub_y1:
                continue

            # Giới hạn tổng số ảnh mỗi tracklet
            if track_counts.get(bt_id, 0) >= MAX_PER_TRACK:
                continue

            # Chống ảnh trùng lặp: track vừa lưu cách đây < MIN_GAP frame thì bỏ qua
            last = track_last_saved.get(bt_id)
            if last is not None and (frame_idx - last) < MIN_GAP:
                continue

            crop = frame[ub_y1:ub_y2, ub_x1:ub_x2]
            if crop.size == 0:
                continue

            # Cấp số track nối tiếp + tạo thư mục
            gid = global_track_id(bt_id)
            track_dir = cam_dir / f"track{gid:04d}"
            track_dir.mkdir(parents=True, exist_ok=True)

            fname = f"f{frame_idx:06d}.jpg"
            fpath = track_dir / fname
            cv2.imwrite(str(fpath), crop)

            track_counts[bt_id] = track_counts.get(bt_id, 0) + 1
            track_last_saved[bt_id] = frame_idx

            rel_path = str(fpath.relative_to(output_dir)).replace("\\", "/")
            meta_rows.append({
                "cam": CAM_ID,
                "track_id": gid,
                "frame": frame_idx,
                "path": rel_path,
            })

        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx}/{total_frames} — "
                  f"{len(track_counts)} track, {len(meta_rows)} ảnh")

    cap.release()

    # --- Xuất metadata CSV (nối tiếp nếu đã có, để tích lũy qua nhiều lần) ---
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cam", "track_id", "frame", "path"])
        if write_header:
            writer.writeheader()
        writer.writerows(meta_rows)

    # --- Thống kê cuối ---
    tracks_ge3 = sum(1 for c in track_counts.values() if c >= 3)
    print(f"\n=== THỐNG KÊ CAM {CAM_ID} (lần chạy này) ===")
    print(f"Track mới phát hiện      : {len(track_counts)}")
    print(f"  trong đó có ≥3 ảnh     : {tracks_ge3}")
    print(f"Ảnh lưu lần này          : {len(meta_rows)}")
    if existing_max:
        print(f"Đánh số từ track{existing_max + 1:04d} → track{existing_max + len(id_map):04d}")
    print(f"Metadata CSV             : {csv_path}")


if __name__ == "__main__":
    main()
