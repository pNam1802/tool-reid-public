# ===================== CẤU HÌNH — server ghi đè khi gọi =====================
VIDEO_PATH = "video/20260613_1530_cam4.mp4"   # đường dẫn video (label_server.py ghi vào)
POS        = 0.1                  # vị trí frame: 0.0 = frame ĐẦU, 1.0 = frame CUỐI
# ============================================================================
#
# Trích 1 frame của video tại vị trí POS (tỉ lệ 0.0–1.0) rồi GHI JPEG RA STDOUT.
# Dùng cho zone editor trên web: lấy đúng frame video thật để vẽ ROI lên.
#
# Quy ước với label_server.py (/frame):
#   - chỉ ghi BYTES JPEG ra stdout, mọi log khác ghi ra stderr;
#   - thoát mã 0 nếu thành công, khác 0 nếu lỗi (server kiểm tra returncode).

import sys

import cv2


def log(*a):
    """In thông báo ra stderr (KHÔNG đụng stdout vì stdout là dữ liệu ảnh)."""
    print(*a, file=sys.stderr, flush=True)


def read_frame_at(cap, pos: float):
    """Đọc frame tại tỉ lệ pos. Trả về (ok, frame). Có nhiều lớp dự phòng."""
    pos = max(0.0, min(1.0, float(pos)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Cách 1: nhảy theo chỉ số frame (chính xác nhất khi biết tổng frame)
    if total > 0:
        target = int(round(pos * (total - 1)))
        target = max(0, min(total - 1, target))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        log(f"[fallback] doc frame #{target} that bai, thu cach khac")

    # Cách 2: nhảy theo tỉ lệ thời lượng (một số codec không cho seek theo frame)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, pos)
    ok, frame = cap.read()
    if ok and frame is not None:
        return True, frame

    # Cách 3: quay về đầu, đọc frame đầu tiên đọc được
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return ok, frame


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log(f"[LOI] Khong mo duoc video: {VIDEO_PATH}")
        sys.exit(1)

    ok, frame = read_frame_at(cap, POS)
    cap.release()

    if not ok or frame is None:
        log(f"[LOI] Khong doc duoc frame nao tu: {VIDEO_PATH}")
        sys.exit(2)

    # Mã hoá JPEG (giữ NGUYÊN kích thước gốc để toạ độ zone khớp với extract_crops)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        log("[LOI] Ma hoa JPEG that bai")
        sys.exit(3)

    h, w = frame.shape[:2]
    log(f"Frame {w}x{h} tai POS={POS} -> {len(buf)} bytes")
    sys.stdout.buffer.write(buf.tobytes())   # CHI ghi bytes anh ra stdout
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
