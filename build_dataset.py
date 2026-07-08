# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
CROPS_DIR   = "crops"        # thư mục output bước 1 (extract_crops.py)
LABELS_CSV  = "labels.csv"   # file CSV đã gán nhãn (từ bước 3)
OUTPUT_DIR  = "myReIDNew"    # thư mục dataset đầu ra (Market-1501 format) — KHÔNG ghi đè myreid thật
TEST_RATIO  = 0.3            # tỉ lệ identity đưa vào test (chỉ với identity có ≥2 camera)
RANDOM_SEED = 42             # seed ngẫu nhiên để kết quả có thể tái tạo
# =================================================================

import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Ép stdout UTF-8 để in tiếng Việt không crash trên terminal cp1252 (Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def get_track_images(crops_dir: Path, row: dict) -> list[Path]:
    """Trả về danh sách ảnh đã sắp xếp của một tracklet.
    Ưu tiên dùng track_dir (batch-aware), fallback về đường dẫn cũ."""
    track_dir_rel = row.get("track_dir", "")
    if track_dir_rel:
        track_dir = crops_dir / track_dir_rel
    else:
        # Tương thích ngược với labels.csv không có cột track_dir
        track_dir = crops_dir / f"cam{row['cam']}" / f"track{int(row['track_id']):04d}"
    if not track_dir.exists():
        return []
    return sorted(track_dir.glob("*.jpg"))


# Đếm số lần (pid, cam, frame) xuất hiện — tránh 2 batch trùng frame ghi đè nhau
_name_counter: dict = defaultdict(int)


def copy_to_dataset(src: Path, dst_dir: Path, pid: int, cam: int, frame_num: int) -> None:
    """Copy 1 ảnh vào thư mục đích với tên chuẩn Market-1501."""
    # Định dạng: {pid:04d}_c{cam}s1_{frame:06d}_{k:02d}.jpg
    key = (str(dst_dir), pid, cam, frame_num)
    _name_counter[key] += 1
    k = _name_counter[key]
    fname = f"{pid:04d}_c{cam}s1_{frame_num:06d}_{k:02d}.jpg"
    shutil.copy2(str(src), str(dst_dir / fname))


def main():
    random.seed(RANDOM_SEED)

    crops_dir  = Path(CROPS_DIR)
    labels_path = Path(LABELS_CSV)
    output_dir  = Path(OUTPUT_DIR)

    if not labels_path.exists():
        print(f"[LỖI] Không tìm thấy file nhãn: {labels_path}")
        sys.exit(1)
    if not crops_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục crops: {crops_dir}")
        sys.exit(1)

    # --- Đọc labels.csv, bỏ tracklet có global_pid == -1 ---
    rows = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = int(row["global_pid"])
            if pid == -1:
                continue
            rows.append({
                "cam":        int(row["cam"]),
                "track_id":   int(row["track_id"]),
                "global_pid": pid,
                "track_dir":  row.get("track_dir", ""),   # batch-aware path
            })

    if not rows:
        print("[LỖI] Không có tracklet hợp lệ (tất cả global_pid == -1).")
        sys.exit(1)

    # --- Nhóm tracklet theo identity ---
    pid_rows: dict[int, list] = defaultdict(list)
    for r in rows:
        pid_rows[r["global_pid"]].append(r)

    # --- Phân loại identity: 1 camera vs ≥2 camera ---
    single_cam_pids = []
    multi_cam_pids  = []
    for pid, prows in pid_rows.items():
        n_cams = len({r["cam"] for r in prows})
        (multi_cam_pids if n_cams >= 2 else single_cam_pids).append(pid)

    if not multi_cam_pids:
        print("[CẢNH BÁO] Không có identity nào xuất hiện ở ≥2 camera!")
        print("  → Tất cả dữ liệu sẽ vào train; tập query/gallery sẽ rỗng.")
        print("  → Hãy gán nhãn thêm tracklet cross-camera trong labels.csv.")

    # --- Chia train/test theo identity ---
    random.shuffle(multi_cam_pids)
    n_test    = max(1, round(len(multi_cam_pids) * TEST_RATIO)) if multi_cam_pids else 0
    test_pids = set(multi_cam_pids[:n_test])
    train_pids = set(single_cam_pids) | set(multi_cam_pids[n_test:])

    # --- Đánh số lại PID liên tiếp từ 0 (chuẩn Market-1501) ---
    all_pids_sorted = sorted(pid_rows.keys())
    pid_remap = {old: new for new, old in enumerate(all_pids_sorted)}

    # --- Tạo cấu trúc thư mục (xóa sạch nếu đã có để tránh ảnh thừa) ---
    train_dir   = output_dir / "bounding_box_train"
    query_dir   = output_dir / "query"
    gallery_dir = output_dir / "bounding_box_test"

    if output_dir.exists():
        existing = sum(1 for d in [train_dir, query_dir, gallery_dir] if d.exists())
        if existing > 0:
            print(f"[CẢNH BÁO] Thư mục {output_dir} đã có dataset cũ — sẽ xóa và build lại.")
            for d in [train_dir, query_dir, gallery_dir]:
                if d.exists():
                    shutil.rmtree(str(d))

    for d in [train_dir, query_dir, gallery_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "query": 0, "gallery": 0}

    # --- Điền tập train ---
    for pid in sorted(train_pids):
        new_pid = pid_remap[pid]
        for row in pid_rows[pid]:
            for img in get_track_images(crops_dir, row):
                frame_num = int(img.stem[1:])   # f{NNNNNN} → NNNNNN
                copy_to_dataset(img, train_dir, new_pid, row["cam"], frame_num)
                stats["train"] += 1

    # --- Điền tập test (query + gallery) ---
    for pid in sorted(test_pids):
        new_pid = pid_remap[pid]

        # Nhóm tracklet theo camera
        cam_rows: dict[int, list] = defaultdict(list)
        for row in pid_rows[pid]:
            cam_rows[row["cam"]].append(row)

        for cam, c_rows in cam_rows.items():
            # Gom tất cả ảnh của identity này trong camera này
            all_images: list[Path] = []
            for row in c_rows:
                all_images.extend(get_track_images(crops_dir, row))
            all_images.sort()

            if not all_images:
                continue

            # 1 ảnh đầu làm query, phần còn lại vào gallery
            query_img = all_images[0]
            copy_to_dataset(query_img, query_dir, new_pid, cam, int(query_img.stem[1:]))
            stats["query"] += 1

            for img in all_images[1:]:
                copy_to_dataset(img, gallery_dir, new_pid, cam, int(img.stem[1:]))
                stats["gallery"] += 1

    # --- Ảnh đại diện (_identities/pid_XXXX.jpg): 1 crop/identity để web hiện thumbnail + đếm đúng ---
    idd = output_dir / "_identities"
    if idd.exists():
        shutil.rmtree(str(idd))
    idd.mkdir(parents=True, exist_ok=True)
    for old_pid in all_pids_sorted:
        new_pid = pid_remap[old_pid]
        for row in pid_rows[old_pid]:
            imgs = get_track_images(crops_dir, row)
            if imgs:
                shutil.copy2(str(imgs[len(imgs) // 2]), str(idd / f"pid_{new_pid:04d}.jpg"))
                break

    # --- In thống kê ---
    print(f"\n=== THỐNG KÊ DATASET ===")
    print(f"Tổng identity          : {len(all_pids_sorted)}")
    print(f"Identity train         : {len(train_pids)}")
    print(f"Identity test          : {len(test_pids)}")
    print(f"Ảnh train              : {stats['train']}")
    print(f"Ảnh query              : {stats['query']}")
    print(f"Ảnh gallery            : {stats['gallery']}")
    print(f"Thư mục output         : {output_dir}/")
    print(f"  ├─ bounding_box_train/  ({stats['train']} ảnh)")
    print(f"  ├─ query/               ({stats['query']} ảnh)")
    print(f"  └─ bounding_box_test/   ({stats['gallery']} ảnh)")


if __name__ == "__main__":
    main()
