# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
VIDEO_DIR    = "video"        # nơi bỏ 3 file mp4 (1 file/camera)
CROPS_DIR    = "crops"        # ảnh crop theo track (TẠM)
MONTAGE_DIR  = "montages"     # ảnh lưới để gán nhãn (TẠM)
LABELS_CSV   = "labels.csv"   # nhãn đợt hiện tại (TẠM)
DATASET_DIR  = "myReIDNew"    # DATASET CHÍNH mới (Market-1501) — tích lũy vĩnh viễn ★
QR_DIR       = "qr"           # folder chứa ảnh chọn tay từ dataset (vd để gắn mã QR) ★
DATASET2_DIR = "myreid2"      # DATASET PHỤ (Import dataset khác vào đây để xem & gộp vào chính) ★

# Ánh xạ số camera: tên file "...camN..." → số Cam thật trong dataset.
# Dùng khi camera vật lý đánh số khác quy ước dataset (vd file cam3/4/5 nhưng
# dataset dùng 1/2/3). Để {} nếu lấy đúng số trong tên file.
CAM_MAP = {3: 1, 4: 2, 5: 3}

# Tham số extract (ghi vào extract_crops.py khi chạy)
SAVE_EVERY    = 8
MIN_HEIGHT    = 80
MAX_PER_TRACK = 40   # nhiều ảnh hơn cho người ở lâu trong video dài
MIN_GAP       = 50   # ~2s/ảnh @25fps → rải mẫu, không dồn hết vào ~24s đầu
# Tham số montage
MONTAGE_MIN_IMAGES = 3
# Tự gợi ý ID bằng osnet2.onnx ngay trong bước crop+montage (True) hay để bấm nút 🤖 riêng (False)
AUTO_SUGGEST  = True
# Tỉ lệ identity (≥2 camera) đưa vào test khi gộp vào dataset
TEST_RATIO    = 0.3
# Người chỉ xuất hiện ở 1 camera (chống camera bias):
#   "distractor" = thả vào gallery làm người nhiễu (đánh giá sát thực tế)
#   "drop"       = bỏ hẳn, không đưa vào dataset
SINGLE_CAM_MODE = "distractor"
DISTRACTOR_BASE = 100000      # pid >= giá trị này là distractor, KHÔNG phải identity train/reuse
ZONES_FILE   = "zones.json"   # lưu zone vẽ tay theo cam_id (tích lũy vĩnh viễn)

HOST         = "127.0.0.1"
PORT         = 8000
OPEN_BROWSER = True
# =================================================================
#
# Cách chạy:   python label_server.py
# Chỉ dùng thư viện chuẩn Python cho server (extract/montage gọi script sẵn có).
#
# LUỒNG:  bỏ video vào video/  →  web: Crop → Montage → Gán ID → Gộp vào dataset
#         Gộp xong: xóa sạch crops/ montages/ labels.csv + video đã xử lý,
#         chỉ giữ lại dataset/ lớn dần.
#

import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
UI_FILE = HERE / "label_ui.html"
PYTHON = sys.executable

# Trạng thái gán nhãn: -2 = chưa xem, -1 = bỏ qua, >=0 = đã gán PID
UNSEEN = -2
SKIP = -1

FIELDS = ["cam", "track_id", "num_images", "montage", "global_pid", "track_dir"]

# ── State trong bộ nhớ ────────────────────────────────────────────────────────
TRACKS: list[dict] = []
BY_DIR: dict[str, dict] = {}
_LOCK = threading.Lock()

# Tiến trình xử lý nền (extract + montage)
PROGRESS = {"running": False, "phase": "", "done": False, "error": ""}
LOG = deque(maxlen=40)

# Khóa gộp dataset: chặn bấm 'Hoàn tất' 2 lần / 2 request cùng lúc → tránh copy lặp ảnh
_COMMIT_LOCK = threading.Lock()
# Cờ: đã gộp XONG nhưng crops/montages còn bị khóa tạm, đang dọn nền → state coi như idle
CLEANUP_PENDING = {"on": False}
# Tiến trình gộp dataset (chạy nền) để UI hiện thanh %; poll qua /api/commit_progress
COMMIT_PROGRESS = {"running": False, "phase": "", "done": 0, "total": 0, "result": None, "error": ""}


# ══════════════════════════════════════════════════════════════════════════════
# Zones (ROI polygon) — lưu theo cam_id (string key vì JSON key luôn là string)
# ══════════════════════════════════════════════════════════════════════════════
def load_zones() -> dict:
    """Load zones.json → dict {"1": [[x,y],...], ...}. Trả {} nếu chưa có."""
    p = HERE / ZONES_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_zones(zones: dict) -> None:
    p = HERE / ZONES_FILE
    p.write_text(json.dumps(zones, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Ghi CONFIG block vào script con (không xâm phạm logic, giống pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════
def write_config(script: str, updates: dict) -> None:
    p = HERE / script
    content = p.read_text(encoding="utf-8")
    for key, value in updates.items():
        if isinstance(value, bool):
            pattern = rf'^({re.escape(key)}\s*=\s*)(?:True|False)'
            new_val = "True" if value else "False"
        elif isinstance(value, str):
            pattern = rf'^({re.escape(key)}\s*=\s*)"[^"]*"'
            new_val = f'"{value}"'
        else:
            pattern = rf'^({re.escape(key)}\s*=\s*)[\d.]+'
            new_val = str(value)
        content = re.sub(pattern, lambda m, v=new_val: m.group(1) + v,
                         content, flags=re.MULTILINE)
    p.write_text(content, encoding="utf-8")


def run_step(script: str) -> bool:
    """Chạy 1 script con, đẩy stdout vào LOG. Trả về True nếu thành công."""
    # Ép tiến trình con in UTF-8 (tránh UnicodeEncodeError do terminal cp1252
    # khi script in tiếng Việt) — sửa 1 lần cho mọi script con.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(
        [PYTHON, "-u", str(HERE / script)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=str(HERE),   # crops/ montages/ luôn nằm cạnh script
        env=env,
    )
    for line in proc.stdout:
        LOG.append(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        LOG.append(f"[LỖI] {script} kết thúc với mã {proc.returncode}")
    return proc.returncode == 0


def _safe_rmtree(path: Path, retries: int = 8, delay: float = 0.3) -> bool:
    """Xóa cây thư mục, chịu được file bị KHÓA TẠM trên Windows.

    Server đa luồng có thể đang đọc ảnh crop (phục vụ /crop, /img, /montage)
    đúng lúc dọn → WinError 5. Ta thử lại vài lần (handle sẽ được nhả) và bỏ
    cờ read-only nếu có. Trả True nếu cuối cùng đã xóa sạch.
    """
    def onerror(func, p, exc):
        try:
            os.chmod(p, 0o666)
            func(p)
        except Exception:
            pass
    for _ in range(retries):
        if not path.exists():
            return True
        shutil.rmtree(str(path), onerror=onerror)
        if not path.exists():
            return True
        time.sleep(delay)
    return not path.exists()


def _retry_cleanup_temp() -> None:
    """Dọn nền crops/montages sau khi gộp, khi file còn bị khóa tạm (Windows).
    Thử lại tới khi sạch rồi tắt cờ CLEANUP_PENDING (state trở về idle đúng)."""
    for _ in range(60):
        time.sleep(1.5)
        ok = True
        if Path(CROPS_DIR).exists():
            ok &= _safe_rmtree(Path(CROPS_DIR))
        if Path(MONTAGE_DIR).exists():
            ok &= _safe_rmtree(Path(MONTAGE_DIR))
        if ok:
            break
    CLEANUP_PENDING["on"] = False


# ══════════════════════════════════════════════════════════════════════════════
# Quét video/ → danh sách camera
# ══════════════════════════════════════════════════════════════════════════════
# Định dạng video chấp nhận (OpenCV/ffmpeg đọc được tất cả các đuôi này)
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}


def _mp4_files(folder: Path) -> list[Path]:
    """Liệt kê file video (mp4/mkv/mov/avi…), không phân biệt hoa/thường, không trùng."""
    if not folder.exists():
        return []
    seen, out = set(), []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            key = str(f).lower()
            if key not in seen:
                seen.add(key)
                out.append(f)
    return out


def scan_videos() -> list[dict]:
    p = Path(VIDEO_DIR)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        return []
    out = []
    for i, f in enumerate(_mp4_files(p)):
        # Tự nhận số Cam từ tên file (vd "..._cam3.mp4" → 3); không có thì theo thứ tự.
        # Giúp cùng 1 camera vật lý luôn cùng số Cam qua mọi đợt nạp → tích lũy đúng.
        m = re.search(r"cam(\d+)", f.name, re.IGNORECASE)
        cam_id = int(m.group(1)) if m else i + 1
        cam_id = CAM_MAP.get(cam_id, cam_id)   # áp ánh xạ camera nếu có
        out.append({"video": str(f).replace("\\", "/"), "cam_id": cam_id, "name": f.name})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Quét crops + montages → labels.csv (mỗi track 1 dòng, pid mặc định = -2)
# ══════════════════════════════════════════════════════════════════════════════
def scan_to_labels() -> None:
    crops_dir   = Path(CROPS_DIR)
    montage_dir = Path(MONTAGE_DIR)
    csv_path    = Path(LABELS_CSV)

    existing: dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row.get("track_dir", "")] = row

    rows: list[dict] = []
    if crops_dir.exists():
        for track_dir in sorted(crops_dir.glob("cam*/track*")):
            images = sorted(track_dir.glob("*.jpg"))
            if not images:
                continue
            m = re.search(r"cam(\d+)", track_dir.parent.name)
            if not m:
                continue
            cam_id   = int(m.group(1))
            track_id = int(track_dir.name.replace("track", ""))
            track_dir_rel = str(track_dir.relative_to(crops_dir)).replace("\\", "/")
            mp = montage_dir / f"{track_dir.parent.name}_{track_dir.name}.jpg"
            montage_rel = str(mp).replace("\\", "/") if mp.exists() else ""
            pid = int(existing[track_dir_rel]["global_pid"]) if track_dir_rel in existing else UNSEEN
            rows.append({
                "cam": cam_id, "track_id": track_id, "num_images": len(images),
                "montage": montage_rel, "global_pid": pid, "track_dir": track_dir_rel,
            })

    rows.sort(key=lambda r: (r["cam"], r["track_dir"]))
    _load_rows(rows)
    save_csv()


def _load_rows(rows: list[dict]) -> None:
    TRACKS.clear(); BY_DIR.clear()
    TRACKS.extend(rows)
    for r in rows:
        BY_DIR[r["track_dir"]] = r


def load_labels_from_disk() -> None:
    """Nạp labels.csv vào bộ nhớ (khi server khởi động giữa chừng)."""
    csv_path = Path(LABELS_CSV)
    if not csv_path.exists():
        _load_rows([])
        return
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "cam": int(row["cam"]), "track_id": int(row["track_id"]),
                "num_images": int(row["num_images"]), "montage": row.get("montage", ""),
                "global_pid": int(row["global_pid"]), "track_dir": row["track_dir"],
            })
    _load_rows(rows)


def save_csv() -> None:
    with _LOCK:
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(TRACKS)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset (Market-1501) — đọc identity đã có, gộp đợt mới, dọn tạm
# ══════════════════════════════════════════════════════════════════════════════
def ds_dirs(base_dir: str = None):
    base = Path(base_dir or DATASET_DIR)
    return (base / "bounding_box_train", base / "query",
            base / "bounding_box_test", base / "_identities")


def existing_identities(base_dir: str = None) -> list[dict]:
    """Danh sách identity đã có trong dataset (để đối chiếu khi gán đợt mới)."""
    _, _, _, idd = ds_dirs(base_dir)
    if not idd.exists():
        return []
    out = []
    for f in sorted(idd.glob("pid_*.jpg")):
        m = re.match(r"pid_(\d+)\.jpg", f.name)
        if m:
            out.append({"pid": int(m.group(1))})
    return out


def _all_dataset_pids() -> set[int]:
    train, query, test, idd = ds_dirs()
    pids = set()
    for d in (train, query, test):
        if d.exists():
            for f in d.glob("*.jpg"):
                mm = re.match(r"(\d+)_c", f.name)
                if mm:
                    pid = int(mm.group(1))
                    if pid < DISTRACTOR_BASE:          # bỏ qua distractor
                        pids.add(pid)
    if idd.exists():
        for f in idd.glob("pid_*.jpg"):
            mm = re.match(r"pid_(\d+)\.jpg", f.name)
            if mm:
                pids.add(int(mm.group(1)))
    return pids


def _max_distractor_pid() -> int:
    """pid distractor lớn nhất đang có trong gallery (để đánh số tiếp)."""
    _, _, test, _ = ds_dirs()
    mx = DISTRACTOR_BASE - 1
    if test.exists():
        for f in test.glob("*.jpg"):
            mm = re.match(r"(\d+)_c", f.name)
            if mm:
                p = int(mm.group(1))
                if p >= DISTRACTOR_BASE:
                    mx = max(mx, p)
    return mx


def camera_coverage() -> dict:
    """Thống kê 'sức khỏe' dataset: mỗi identity xuất hiện ở mấy camera."""
    train, query, test, _ = ds_dirs()
    pid_cams: dict[int, set] = defaultdict(set)
    distractor_pids: set = set()
    distractor_imgs = 0
    for d in (train, query, test):
        if not d.exists():
            continue
        for f in d.glob("*.jpg"):
            mm = re.match(r"(\d+)_c(\d+)s", f.name)
            if not mm:
                continue
            pid, cam = int(mm.group(1)), int(mm.group(2))
            if pid >= DISTRACTOR_BASE:
                distractor_pids.add(pid)
                distractor_imgs += 1
            else:
                pid_cams[pid].add(cam)
    by_cams = {1: 0, 2: 0, 3: 0, "4+": 0}
    for cams in pid_cams.values():
        n = len(cams)
        by_cams["4+" if n >= 4 else n] = by_cams.get("4+" if n >= 4 else n, 0) + 1
    return {
        "identities": len(pid_cams),
        "by_cams": by_cams,
        "ge2": sum(1 for c in pid_cams.values() if len(c) >= 2),
        "distractor_people": len(distractor_pids),
        "distractor_images": distractor_imgs,
    }


def next_pid() -> int:
    used = _all_dataset_pids()
    batch = {t["global_pid"] for t in TRACKS if t["global_pid"] >= 0}
    allp = used | batch
    return (max(allp) + 1) if allp else 0


def free_pids() -> list[int]:
    """Các số ID bị TRỐNG (đã xóa trong lúc lọc) nằm giữa dãy ID thật hiện có → tái dùng khi gán.

    = các lỗ trong khoảng [min, max) của tập (ID thật trong dataset ∪ ID đã gán trong đợt này).
    Bỏ qua distractor (pid ≥ DISTRACTOR_BASE). Cùng nguồn với next_pid() nên luôn nhất quán:
    số đã dùng (kể cả vừa gán trong đợt) sẽ không xuất hiện → không lỡ gán 1 số cho 2 người.
    """
    used = _all_dataset_pids()
    used |= {t["global_pid"] for t in TRACKS if t["global_pid"] >= 0}
    used = {p for p in used if 0 <= p < DISTRACTOR_BASE}   # distractor không tính vào dãy ID thật
    if len(used) < 2:
        return []
    lo, hi = min(used), max(used)
    return [p for p in range(lo, hi) if p not in used]


def _identity_split(pid: int) -> str | None:
    """Xác định identity đã thuộc train hay test (None nếu là người mới)."""
    train, query, test, _ = ds_dirs()
    pref = f"{pid:04d}_c"
    for d in (test, query):
        if d.exists() and any(f.name.startswith(pref) for f in d.glob("*.jpg")):
            return "test"
    if train.exists() and any(f.name.startswith(pref) for f in train.glob("*.jpg")):
        return "train"
    return None


def _cams_with_query(pid: int) -> set[int]:
    _, query, _, _ = ds_dirs()
    cams = set()
    if query.exists():
        for f in query.glob(f"{pid:04d}_c*.jpg"):
            mm = re.search(r"_c(\d+)s", f.name)
            if mm:
                cams.add(int(mm.group(1)))
    return cams


def _market_copy(src: Path, dst_dir: Path, pid: int, cam: int, frame: int) -> None:
    k = 1
    while True:
        name = f"{pid:04d}_c{cam}s1_{frame:06d}_{k:02d}.jpg"
        dst = dst_dir / name
        if not dst.exists():
            break
        k += 1
    shutil.copy2(str(src), str(dst))


def _frame_of(img: Path) -> int:
    m = re.search(r"f(\d+)", img.stem)
    return int(m.group(1)) if m else 0


def commit_to_dataset() -> dict:
    """Gộp các track đã gán PID vào dataset/, rồi xóa sạch dữ liệu tạm."""
    random.seed()
    train, query, test, idd = ds_dirs()
    for d in (train, query, test, idd):
        d.mkdir(parents=True, exist_ok=True)

    crops = Path(CROPS_DIR)
    montages = Path(MONTAGE_DIR)

    # Gom track theo PID (chỉ lấy pid >= 0)
    by_pid: dict[int, list[dict]] = defaultdict(list)
    for t in TRACKS:
        if t["global_pid"] >= 0:
            by_pid[t["global_pid"]].append(t)

    stats = {"new_ids": 0, "updated_ids": 0, "train": 0, "query": 0, "gallery": 0,
             "new_distractor": 0, "distractor_images": 0, "dropped": 0}
    next_distractor = _max_distractor_pid() + 1
    # Đếm tổng ảnh cần copy → cho UI hiện thanh tiến trình khi gộp
    total_imgs = sum(len(list((crops / t["track_dir"]).glob("*.jpg")))
                     for t in TRACKS if t["global_pid"] >= 0)
    COMMIT_PROGRESS.update(phase="Đang gộp ảnh vào dataset", total=total_imgs, done=0)

    for pid, group in sorted(by_pid.items()):
        split = _identity_split(pid)
        is_new = split is None
        cams = {t["cam"] for t in group}

        # CÁCH B — người MỚI chỉ 1 camera → distractor (chống camera bias, lỗi #2/#7)
        if is_new and len(cams) == 1:
            if SINGLE_CAM_MODE == "drop":
                stats["dropped"] += 1
                continue
            dpid = next_distractor
            next_distractor += 1
            stats["new_distractor"] += 1
            for t in group:
                for img in sorted((crops / t["track_dir"]).glob("*.jpg")):
                    _market_copy(img, test, dpid, t["cam"], _frame_of(img))   # vào gallery
                    stats["distractor_images"] += 1; COMMIT_PROGRESS["done"] += 1
            continue   # không lưu avatar, không vào train, không phải identity

        if is_new:
            split = "test" if (len(cams) >= 2 and random.random() < TEST_RATIO) else "train"
            stats["new_ids"] += 1
            # Lưu ảnh đại diện = 1 ẢNH CROP ĐƠN (ảnh giữa của track đầu) cho dễ nhận mặt
            rep_imgs = sorted((crops / group[0]["track_dir"]).glob("*.jpg"))
            if rep_imgs:
                shutil.copy2(str(rep_imgs[len(rep_imgs) // 2]), str(idd / f"pid_{pid:04d}.jpg"))
        else:
            stats["updated_ids"] += 1

        if split == "train":
            for t in group:
                folder = crops / t["track_dir"]
                for img in sorted(folder.glob("*.jpg")):
                    _market_copy(img, train, pid, t["cam"], _frame_of(img))
                    stats["train"] += 1; COMMIT_PROGRESS["done"] += 1
        else:  # test
            qcams = _cams_with_query(pid)
            # Gom ảnh theo camera
            cam_imgs: dict[int, list[Path]] = defaultdict(list)
            for t in group:
                folder = crops / t["track_dir"]
                for img in sorted(folder.glob("*.jpg")):
                    cam_imgs[t["cam"]].append(img)
            for cam, imgs in cam_imgs.items():
                imgs.sort()
                start = 0
                if cam not in qcams and imgs:
                    _market_copy(imgs[0], query, pid, cam, _frame_of(imgs[0]))
                    stats["query"] += 1; COMMIT_PROGRESS["done"] += 1
                    start = 1
                for img in imgs[start:]:
                    _market_copy(img, test, pid, cam, _frame_of(img))
                    stats["gallery"] += 1; COMMIT_PROGRESS["done"] += 1

    # ── Đợt đã copy xong → ĐÁNH DẤU ĐÃ GỘP NGAY (chống bấm 'Hoàn tất' lần nữa bị copy lặp) ──
    # LỖI CŨ: chỉ clear bộ nhớ KHI dọn crops thành công. Trên Windows nếu crops đang bị
    # khóa (server đa luồng phục vụ /crop /montage /img) thì dọn LỖI → labels.csv + bộ nhớ
    # còn nguyên → UI vẫn thấy đợt cũ → bấm lại 'Hoàn tất' = COPY LẶP toàn bộ ảnh (cộng dồn
    # vô hạn, mỗi ảnh thêm hậu tố _k mới nên không ghi đè).
    # SỬA: xóa bộ nhớ + labels.csv NGAY sau khi copy (bất kể dọn crops được hay chưa) → lần
    # commit kế tiếp không còn track nào để copy → KHÔNG THỂ cộng dồn.
    COMMIT_PROGRESS.update(phase="Đang dọn dữ liệu tạm (crops/montages)…", done=total_imgs)
    _load_rows([])
    if Path(LABELS_CSV).exists():
        try:
            Path(LABELS_CSV).unlink()
        except OSError:
            pass
    cleanup_ok = True
    if crops.exists():
        cleanup_ok &= _safe_rmtree(crops)
    if montages.exists():
        cleanup_ok &= _safe_rmtree(montages)
    if not cleanup_ok:
        # crops/montages còn bị khóa tạm → dọn NỀN; state vẫn coi là idle (đợt đã gộp xong)
        CLEANUP_PENDING["on"] = True
        threading.Thread(target=_retry_cleanup_temp, daemon=True).start()
        print("[CẢNH BÁO] Đã gộp vào dataset XONG; crops/montages đang bị khóa tạm, "
              "sẽ tự dọn nền — KHÔNG cần bấm lại 'Hoàn tất'.")
    stats["cleanup_ok"] = cleanup_ok
    # KHÔNG xóa video ở đây: video đã crop đã bị xóa ngay sau bước crop
    # (xem run_pipeline). Đoạn nào còn trong video/ là CHƯA xử lý → giữ lại
    # cho phiên sau, tránh mất dữ liệu chưa gán.

    stats["coverage"] = camera_coverage()
    stats["dataset"] = dataset_stats()

    # ── In thống kê đợt này ra console ──
    cov = stats["coverage"]
    bc = cov["by_cams"]
    print("\n========== THỐNG KÊ ĐỢT VỪA GỘP ==========")
    print(f"Người mới (≥2 camera, làm identity) : {stats['new_ids']}")
    print(f"Người mới 1 camera → distractor     : {stats['new_distractor']} "
          f"({stats['distractor_images']} ảnh vào gallery)")
    if stats["dropped"]:
        print(f"Người mới 1 camera → bỏ             : {stats['dropped']}")
    print(f"Identity cũ được bổ sung            : {stats['updated_ids']}")
    print(f"Ảnh thêm: {stats['train']} train · {stats['query']} query · {stats['gallery']} gallery")
    print("---------- SỨC KHỎE DATASET ----------")
    print(f"Tổng identity (train+test) : {cov['identities']}")
    print(f"  ở 1 camera : {bc.get(1,0)}   ← càng nhiều càng dễ camera-bias")
    print(f"  ở 2 camera : {bc.get(2,0)}")
    print(f"  ở 3 camera : {bc.get(3,0)}")
    print(f"  ở ≥4 camera: {bc.get('4+',0)}")
    print(f"Identity ≥2 camera (phần 'khỏe' cho ReID): {cov['ge2']} / {cov['identities']}")
    print(f"Distractor trong gallery : {cov['distractor_people']} người, {cov['distractor_images']} ảnh")
    print("==========================================\n")
    return stats


def dataset_stats(base_dir: str = None) -> dict:
    train, query, test, idd = ds_dirs(base_dir)
    def n(d): return len(list(d.glob("*.jpg"))) if d.exists() else 0
    return {
        "identities": len(existing_identities(base_dir)),
        "train": n(train), "query": n(query), "gallery": n(test),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dataset phụ — xem & gộp vào myreid (cấp ID MỚI, không tái dùng)
# ══════════════════════════════════════════════════════════════════════════════
def dataset2_present() -> bool:
    """Có dataset phụ (Market-1501) với ít nhất 1 ảnh không?"""
    train, query, test, _ = ds_dirs(DATASET2_DIR)
    return any(d.exists() and any(d.glob("*.jpg")) for d in (train, query, test))


def _market_frame(name: str) -> int:
    """Lấy số frame từ tên file Market (…_cXs1_<frame>_kk.jpg). 0 nếu không khớp."""
    mm = re.match(r"\d+_c\d+s\d+_(\d+)_", name)
    return int(mm.group(1)) if mm else 0


def merge_dataset2() -> dict:
    """Gộp toàn bộ dataset phụ (myreid2) vào myreid: mỗi ID phụ được cấp 1 ID MỚI
    không trùng ID đã có (đúng luật 'ID mới không tái dùng'), giữ nguyên split
    train/query/gallery; distractor (pid≥DISTRACTOR_BASE) cấp số distractor mới.
    KHÔNG xóa dataset phụ sau khi gộp.
    """
    train2, query2, test2, idd2 = ds_dirs(DATASET2_DIR)
    train1, query1, test1, idd1 = ds_dirs(DATASET_DIR)
    for d in (train1, query1, test1, idd1):
        d.mkdir(parents=True, exist_ok=True)

    src_dirs = {"train": train2, "query": query2, "gallery": test2}
    dst_dirs = {"train": train1, "query": query1, "gallery": test1}

    # Gom ảnh dataset phụ theo pid → {split: [(file, cam), ...]}
    pid_imgs: dict[int, dict] = defaultdict(
        lambda: {"train": [], "query": [], "gallery": []})
    for split, d in src_dirs.items():
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jpg")):
            mm = re.match(r"(\d+)_c(\d+)s", f.name)
            if not mm:
                continue
            pid_imgs[int(mm.group(1))][split].append((f, int(mm.group(2))))

    # Cấp ID mới: tiếp nối ID thật / distractor đang có ở myreid
    used = _all_dataset_pids()
    next_real = max([p for p in used if p < DISTRACTOR_BASE], default=-1) + 1
    next_dis = _max_distractor_pid() + 1

    stats = {"new_ids": 0, "new_distractor": 0,
             "train": 0, "query": 0, "gallery": 0, "images": 0}
    pid_map: dict[int, int] = {}
    for old_pid in sorted(pid_imgs):
        is_dis = old_pid >= DISTRACTOR_BASE
        if is_dis:
            new_pid = next_dis; next_dis += 1; stats["new_distractor"] += 1
        else:
            new_pid = next_real; next_real += 1; stats["new_ids"] += 1
        pid_map[old_pid] = new_pid
        for split in ("train", "query", "gallery"):
            for f, cam in pid_imgs[old_pid][split]:
                _market_copy(f, dst_dirs[split], new_pid, cam, _market_frame(f.name))
                stats[split] += 1
                stats["images"] += 1
        if not is_dis:                                   # chép avatar nếu có
            av = idd2 / f"pid_{old_pid:04d}.jpg"
            if av.exists():
                shutil.copy2(str(av), str(idd1 / f"pid_{new_pid:04d}.jpg"))

    stats["dataset"] = dataset_stats()
    stats["pid_map"] = {str(k): v for k, v in pid_map.items()}
    return stats


def _pick_folder():
    """Mở hộp thoại chọn folder NATIVE trên máy chủ (= máy người dùng, vì app local).
    Trả: đường dẫn (str) nếu chọn · "" nếu bấm Hủy · None nếu không mở được tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)   # đưa hộp thoại lên trước trình duyệt
        p = filedialog.askdirectory(
            title="Chọn folder dataset phụ (bounding_box_train hoặc Market root)")
        root.destroy()
        return p or ""
    except Exception:
        return None


def ds2_import(src_path: str) -> dict:
    """Import 1 folder vào dataset phụ (myreid2) để xem trước khi gộp.
    Chấp nhận: (a) folder gốc Market (có bounding_box_train/query/...),
               (b) 1 folder ảnh phẳng (vd chính là bounding_box_train) → coi là train.
    THAY THẾ dataset phụ hiện có. KHÔNG đụng dataset chính.
    """
    src = Path(str(src_path).strip().strip('"').strip("'"))
    if not src.exists() or not src.is_dir():
        return {"ok": False, "error": "Đường dẫn không tồn tại hoặc không phải thư mục"}

    dst_root = Path(DATASET2_DIR)
    if dst_root.resolve() == src.resolve():
        return {"ok": False, "error": "Không thể import chính thư mục myreid2"}
    if dst_root.exists():                     # thay thế dataset phụ cũ
        _safe_rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    if (src / "bounding_box_train").exists():           # (a) Market root
        for sub in ("bounding_box_train", "query", "bounding_box_test", "_identities"):
            s = src / sub
            if not s.exists():
                continue
            d = dst_root / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in s.glob("*.jpg"):
                shutil.copy2(str(f), str(d / f.name)); copied += 1
    else:                                               # (b) folder ảnh phẳng → train
        d = dst_root / "bounding_box_train"
        d.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.jpg"):
            shutil.copy2(str(f), str(d / f.name)); copied += 1

    if copied == 0:
        _safe_rmtree(dst_root)
        return {"ok": False, "error": "Không thấy ảnh .jpg nào trong thư mục"}
    return {"ok": True, "copied": copied, "stats": dataset_stats(DATASET2_DIR)}


# ══════════════════════════════════════════════════════════════════════════════
# Trình duyệt dataset — liệt kê identity, xem/sửa/xóa ảnh trong myreid
# ══════════════════════════════════════════════════════════════════════════════
def _split_dir(split: str, base_dir: str = None) -> Path | None:
    """Map tên split của UI ('train'/'query'/'gallery') → thư mục dataset."""
    train, query, test, _ = ds_dirs(base_dir)
    return {"train": train, "query": query, "gallery": test}.get(split)


def dataset_identities(base_dir: str = None) -> list[dict]:
    """Tất cả identity trong dataset kèm thống kê: số ảnh mỗi split, số camera."""
    train, query, test, idd = ds_dirs(base_dir)
    splits = {"train": train, "query": query, "gallery": test}
    info: dict[int, dict] = {}
    for split, d in splits.items():
        if not d.exists():
            continue
        for f in d.glob("*.jpg"):
            mm = re.match(r"(\d+)_c(\d+)s", f.name)
            if not mm:
                continue
            pid, cam = int(mm.group(1)), int(mm.group(2))
            rec = info.setdefault(pid, {"train": 0, "query": 0, "gallery": 0,
                                        "cams": set(), "thumb": None})
            rec[split] += 1
            rec["cams"].add(cam)
            if rec["thumb"] is None:
                rec["thumb"] = {"split": split, "name": f.name}

    avatars: set[int] = set()
    if idd.exists():
        for f in idd.glob("pid_*.jpg"):
            mm = re.match(r"pid_(\d+)\.jpg", f.name)
            if mm:
                pid = int(mm.group(1))
                avatars.add(pid)
                info.setdefault(pid, {"train": 0, "query": 0, "gallery": 0,
                                      "cams": set(), "thumb": None})

    out = []
    for pid, rec in info.items():
        out.append({
            "pid": pid,
            "train": rec["train"], "query": rec["query"], "gallery": rec["gallery"],
            "total": rec["train"] + rec["query"] + rec["gallery"],
            "cams": sorted(rec["cams"]), "ncams": len(rec["cams"]),
            "has_avatar": pid in avatars, "thumb": rec["thumb"],
            "distractor": pid >= DISTRACTOR_BASE,
        })
    out.sort(key=lambda r: r["pid"])
    return out


def identity_images(pid: int, base_dir: str = None) -> list[dict]:
    """Tất cả ảnh của 1 pid trong dataset, kèm split + cam (để hiện và chọn xóa)."""
    splits = {"train": _split_dir("train", base_dir), "query": _split_dir("query", base_dir),
              "gallery": _split_dir("gallery", base_dir)}
    out = []
    for split, d in splits.items():
        if not d or not d.exists():
            continue
        for f in sorted(d.glob(f"{pid:04d}_c*.jpg")):
            mm = re.match(r"(\d+)_c(\d+)s", f.name)
            if not mm or int(mm.group(1)) != pid:
                continue
            out.append({"split": split, "name": f.name, "cam": int(mm.group(2))})
    return out


def ds_delete_images(images: list[dict]) -> int:
    """Xóa các ảnh dataset đã chọn. Mỗi phần tử {split, name}. Trả về số ảnh đã xóa."""
    deleted = 0
    for item in images:
        d = _split_dir(item.get("split", ""))
        name = Path(item.get("name", "")).name
        if not d or not name.endswith(".jpg"):
            continue
        f = d / name
        if f.exists():
            f.unlink()
            deleted += 1
    return deleted


def ds_delete_identity(pid: int, base_dir: str = None) -> int:
    """Xóa hẳn 1 identity: mọi ảnh train/query/gallery + ảnh đại diện."""
    _, _, _, idd = ds_dirs(base_dir)
    deleted = 0
    for split in ("train", "query", "gallery"):
        d = _split_dir(split, base_dir)
        if d and d.exists():
            for f in d.glob(f"{pid:04d}_c*.jpg"):
                mm = re.match(r"(\d+)_c", f.name)
                if mm and int(mm.group(1)) == pid:
                    f.unlink()
                    deleted += 1
    avatar = idd / f"pid_{pid:04d}.jpg"
    if avatar.exists():
        avatar.unlink()
    return deleted


def ds_rename_identity(old_pid: int, new_pid: int, base_dir: str = None) -> dict:
    """Đổi ID của 1 identity.
    • new_pid CHƯA có  → đổi số (renumber), giữ nguyên split.
    • new_pid ĐÃ có    → GỘP ảnh của old vào new, đặt vào split của đích để tránh
      rò rỉ train↔test (đích có train → train; đích chỉ test → gallery).
    """
    if old_pid == new_pid:
        return {"ok": False, "error": "ID mới trùng ID cũ"}
    train, query, test, idd = ds_dirs(base_dir)
    split_dirs = {"train": train, "query": query, "gallery": test}

    src = identity_images(old_pid, base_dir)
    if not src:
        return {"ok": False, "error": f"Không thấy ID {old_pid}"}
    tgt = identity_images(new_pid, base_dir)
    merging = bool(tgt)

    if merging:
        tgt_has_train = any(i["split"] == "train" for i in tgt)
        moved = 0
        for im in src:
            f = split_dirs[im["split"]] / im["name"]
            if not f.exists():
                continue
            dst_split = "train" if tgt_has_train else "gallery"
            _market_copy(f, split_dirs[dst_split], new_pid, im["cam"], _market_frame(im["name"]))
            f.unlink(); moved += 1
        av = idd / f"pid_{old_pid:04d}.jpg"          # đích đã có avatar riêng → bỏ avatar nguồn
        if av.exists():
            av.unlink()
        return {"ok": True, "merged": True, "moved": moved,
                "dataset": dataset_stats(base_dir)}

    # Đổi số: đổi tên tại chỗ, giữ nguyên split
    renamed = 0
    for im in src:
        sdir = split_dirs[im["split"]]
        f = sdir / im["name"]
        if not f.exists():
            continue
        _market_copy(f, sdir, new_pid, im["cam"], _market_frame(im["name"]))
        f.unlink(); renamed += 1
    av = idd / f"pid_{old_pid:04d}.jpg"
    if av.exists():
        av.rename(idd / f"pid_{new_pid:04d}.jpg")
    return {"ok": True, "merged": False, "renamed": renamed,
            "dataset": dataset_stats(base_dir)}


def _next_real_pid(base_dir: str = None) -> int:
    """ID thật lớn nhất trong dataset + 1 (bỏ distractor). Dùng khi tách ra ID mới."""
    train, query, test, idd = ds_dirs(base_dir)
    mx = -1
    for d in (train, query, test):
        if d.exists():
            for f in d.glob("*.jpg"):
                m = re.match(r"(\d+)_c", f.name)
                if m:
                    p = int(m.group(1))
                    if p < DISTRACTOR_BASE:
                        mx = max(mx, p)
    if idd.exists():
        for f in idd.glob("pid_*.jpg"):
            m = re.match(r"pid_(\d+)", f.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def ds_split_images(images: list[dict], base_dir: str = None) -> dict:
    """Tách các ảnh đã chọn ra 1 ID MỚI (khỏi ID cũ).
    Dùng để sửa ID bị TRỘN 2 người: chọn ảnh của người thứ 2 → tách ra ID mới.
    Đổi tên file sang pid mới (cùng split), tạo avatar cho ID mới.
    """
    new_pid = _next_real_pid(base_dir)
    _, _, _, idd = ds_dirs(base_dir)
    moved = 0
    first_moved = None
    for item in images:
        d = _split_dir(item.get("split", ""), base_dir)
        name = Path(item.get("name", "")).name
        if not d or not name.endswith(".jpg"):
            continue
        src = d / name
        if not src.exists():
            continue
        mm = re.match(r"\d+_c(\d+)s\d+_(\d+)_", name)
        cam = int(mm.group(1)) if mm else 1
        frame = int(mm.group(2)) if mm else 0
        _market_copy(src, d, new_pid, cam, frame)     # tạo bản tên pid MỚI trong cùng split
        src.unlink()                                   # xóa bản pid cũ
        moved += 1
        if first_moved is None:
            first_moved = (d, cam)
    # avatar cho ID mới (1 ảnh vừa tách) để web hiện thumbnail
    if moved and first_moved is not None:
        d0 = first_moved[0]
        cand = sorted(d0.glob(f"{new_pid:04d}_c*.jpg"))
        if cand:
            idd.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(cand[0]), str(idd / f"pid_{new_pid:04d}.jpg"))
    return {"new_pid": new_pid, "moved": moved}


def ds_set_avatar(pid: int, split: str, name: str) -> bool:
    """Đặt 1 ảnh trong dataset làm ảnh đại diện của identity."""
    d = _split_dir(split)
    name = Path(name).name
    _, _, _, idd = ds_dirs()
    if not d or not name.endswith(".jpg") or not (d / name).exists():
        return False
    idd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(d / name), str(idd / f"pid_{pid:04d}.jpg"))
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Folder QR — gom ảnh chọn tay từ dataset (SAO CHÉP, gộp chung 1 folder phẳng)
# ══════════════════════════════════════════════════════════════════════════════
def _safe_folder_name(name: str) -> str:
    """Tên folder an toàn: chỉ basename, chỉ chữ/số/khoảng trắng/_-. (chống path traversal)."""
    name = Path(str(name)).name.strip()
    name = re.sub(r"[^0-9A-Za-z _.\-]", "_", name)
    return name[:80]


def qr_folders() -> list[dict]:
    """Danh sách folder QR đang có + số ảnh trong mỗi folder."""
    base = Path(QR_DIR)
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            out.append({"name": d.name, "count": len(list(d.glob("*.jpg")))})
    return out


def qr_create_folder(name: str) -> str | None:
    """Tạo 1 folder QR rỗng. Trả tên đã chuẩn hóa, hoặc None nếu tên rỗng."""
    name = _safe_folder_name(name)
    if not name:
        return None
    (Path(QR_DIR) / name).mkdir(parents=True, exist_ok=True)
    return name


def qr_add_images(folder: str, images: list[dict]) -> int:
    """SAO CHÉP các ảnh dataset đã chọn (mỗi phần tử {split, name}) vào folder QR.

    Giữ nguyên tên file (đã có pid ở đầu nên vẫn biết của ai). Nếu trùng tên thì
    thêm hậu tố _k để không ghi đè.
    """
    folder = _safe_folder_name(folder)
    if not folder:
        return 0
    dst_dir = Path(QR_DIR) / folder
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in images:
        d = _split_dir(item.get("split", ""))
        name = Path(item.get("name", "")).name
        if not d or not name.endswith(".jpg"):
            continue
        src = d / name
        if not src.exists():
            continue
        dst = dst_dir / name
        if dst.exists():                       # trùng tên (vd cùng tên ở split khác) → thêm hậu tố
            k = 1
            while (dst_dir / f"{dst.stem}_{k}{dst.suffix}").exists():
                k += 1
            dst = dst_dir / f"{dst.stem}_{k}{dst.suffix}"
        shutil.copy2(str(src), str(dst))
        copied += 1
    return copied


# ══════════════════════════════════════════════════════════════════════════════
# Máy trạng thái
# ══════════════════════════════════════════════════════════════════════════════
def detect_state() -> str:
    if PROGRESS["running"]:
        return "processing"
    crops = Path(CROPS_DIR)
    has_crops = crops.exists() and any(crops.glob("cam*/track*"))
    has_labels = Path(LABELS_CSV).exists()

    if has_labels:
        load_labels_from_disk()
        if any(t["global_pid"] == UNSEEN for t in TRACKS):
            return "label"
        if any(t["global_pid"] >= 0 for t in TRACKS):
            return "commit"
        return "commit"  # tất cả bỏ qua → vẫn cho commit (sẽ không thêm gì)
    if has_crops and not CLEANUP_PENDING["on"]:
        return "montage_pending"   # có crop nhưng chưa montage (server tắt giữa chừng)
    # Sạch (hoặc crops đang được dọn nền sau khi gộp) → cho thêm video
    return "idle"


def label_stats() -> dict:
    total   = len(TRACKS)
    labeled = sum(1 for t in TRACKS if t["global_pid"] >= 0)
    skipped = sum(1 for t in TRACKS if t["global_pid"] == SKIP)
    unseen  = sum(1 for t in TRACKS if t["global_pid"] == UNSEEN)
    n_pids  = len({t["global_pid"] for t in TRACKS if t["global_pid"] >= 0})
    return {"total": total, "labeled": labeled, "skipped": skipped,
            "unseen": unseen, "pids": n_pids}


# ══════════════════════════════════════════════════════════════════════════════
# Chạy pipeline nền: extract (mỗi camera) → montage → scan_to_labels
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(cameras: list[dict]) -> None:
    PROGRESS.update(running=True, phase="extract", done=False, error="")
    LOG.clear()
    zones = load_zones()
    try:
        for cam in cameras:
            cam_id_str = str(cam["cam_id"])
            zone_pts = zones.get(cam_id_str, [])
            if zone_pts:
                LOG.append(f"=== CROP cam {cam['cam_id']}: {cam['video']} (có zone: {len(zone_pts)} điểm) ===")
            else:
                LOG.append(f"=== CROP cam {cam['cam_id']}: {cam['video']} (không có zone) ===")
            write_config("extract_crops.py", {
                "VIDEO_PATH": cam["video"], "CAM_ID": int(cam["cam_id"]),
                "OUTPUT_DIR": CROPS_DIR, "RESET_CAM": False,
                "SAVE_EVERY": SAVE_EVERY, "MIN_HEIGHT": MIN_HEIGHT,
                "MAX_PER_TRACK": MAX_PER_TRACK, "MIN_GAP": MIN_GAP,
                "ZONE_POINTS": zone_pts,
            })
            if not run_step("extract_crops.py"):
                raise RuntimeError(f"Extract cam {cam['cam_id']} lỗi")
            # Crop xong → xóa video đó. Nhờ vậy lần "Nạp thêm video" sau,
            # video/ chỉ còn đoạn CHƯA xử lý → không bao giờ crop lại đoạn cũ.
            try:
                Path(cam["video"]).unlink()
                LOG.append(f"Đã xóa video đã xử lý: {cam['video']}")
            except OSError as e:
                LOG.append(f"[CẢNH BÁO] không xóa được video {cam['video']}: {e}")

        PROGRESS["phase"] = "montage"
        LOG.append("=== MONTAGE ===")
        write_config("make_montages.py", {
            "CROPS_DIR": CROPS_DIR, "OUTPUT_DIR": MONTAGE_DIR,
            "MIN_IMAGES": MONTAGE_MIN_IMAGES,
        })
        if not run_step("make_montages.py"):
            raise RuntimeError("Montage lỗi")

        LOG.append("=== TẠO DANH SÁCH GÁN NHÃN ===")
        scan_to_labels()
        LOG.append(f"Xong! {len(TRACKS)} track sẵn sàng để gán nhãn.")

        # --- Auto gợi ý ID bằng osnet2 (best-effort: lỗi thì giữ nhãn trống, không hỏng cả đợt) ---
        if AUTO_SUGGEST:
            PROGRESS["phase"] = "suggest"
            LOG.append("=== GỢI Ý ID (osnet2.onnx) ===")
            if run_step("suggest_ids_onnx.py"):
                load_labels_from_disk()
                LOG.append(f"Đã gợi ý ID cho {len(TRACKS)} track.")
            else:
                LOG.append("[CẢNH BÁO] Gợi ý ID lỗi — nhãn để trống, bấm 🤖 Gợi ý ID sau hoặc gán tay.")

        PROGRESS.update(phase="done", done=True)
    except Exception as e:
        PROGRESS["error"] = str(e)
        LOG.append(f"[LỖI] {e}")
    finally:
        PROGRESS["running"] = False


# ══════════════════════════════════════════════════════════════════════════════
# HTTP handler
# ══════════════════════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    # Lỗi khi client tự ngắt kết nối (vd kéo slider làm hủy request cũ) — bỏ qua,
    # không phải lỗi server, không cần in traceback.
    _CONN_ERRS = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except self._CONN_ERRS:
            pass

    def _bytes(self, data: bytes, ctype: str, code=200, no_cache=False):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if no_cache:   # buộc trình duyệt tải lại HTML mới, tránh kẹt bản cũ trong cache
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)
        except self._CONN_ERRS:
            pass

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # --- GET ---
    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)

        if path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return

        if path == "/":
            if not UI_FILE.exists():
                self._json({"error": f"Thiếu {UI_FILE.name}"}, 500); return
            self._bytes(UI_FILE.read_bytes(), "text/html; charset=utf-8", no_cache=True); return

        if path == "/api/state":
            self._state_payload(); return

        if path == "/api/progress":
            self._json({**PROGRESS, "log": list(LOG)}); return

        if path == "/api/commit_progress":
            self._json(COMMIT_PROGRESS); return

        if path == "/montage":
            td = qs.get("track_dir", [""])[0]
            rec = BY_DIR.get(td)
            if rec and rec["montage"] and Path(rec["montage"]).exists():
                self._bytes(Path(rec["montage"]).read_bytes(), "image/jpeg"); return
            self._json({"error": "no montage"}, 404); return

        if path == "/frame":
            # Trả về 1 frame JPEG từ video (dùng cho zone editor)
            # Query: ?video=<path>&pos=<0.0-1.0>
            video = qs.get("video", [""])[0]
            pos   = qs.get("pos", ["0.1"])[0]
            if not video:
                self._json({"error": "no video"}, 400); return
            try:
                pos_f = max(0.0, min(1.0, float(pos)))
            except ValueError:
                pos_f = 0.1
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            # Ghi config extract_frame.py
            write_config("extract_frame.py", {"VIDEO_PATH": video, "POS": pos_f})
            proc = subprocess.Popen(
                [PYTHON, "-u", str(HERE / "extract_frame.py")],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(HERE), env=env,
            )
            frame_bytes, err = proc.communicate()
            if proc.returncode != 0 or not frame_bytes:
                self._json({"error": "frame extract failed", "detail": err.decode(errors='replace')}, 500); return
            self._bytes(frame_bytes, "image/jpeg"); return

        if path == "/crop":
            # Ảnh dự phòng: track chưa có montage (ít ảnh) vẫn xem được 1 crop
            td = qs.get("track_dir", [""])[0]
            folder = Path(CROPS_DIR) / td
            imgs = sorted(folder.glob("*.jpg")) if folder.exists() else []
            if imgs:
                self._bytes(imgs[len(imgs) // 2].read_bytes(), "image/jpeg"); return
            self._json({"error": "no crop"}, 404); return

        if path == "/track_images":
            # Danh sách tên ảnh của 1 track (để hiện từng ảnh + cho xóa lẻ)
            td = qs.get("track_dir", [""])[0]
            folder = Path(CROPS_DIR) / td
            names = [f.name for f in sorted(folder.glob("*.jpg"))] if folder.exists() else []
            self._json({"images": names}); return

        if path == "/img":
            # Phục vụ 1 ảnh crop cụ thể (name chỉ lấy phần tên file, chống path traversal)
            td = qs.get("track_dir", [""])[0]
            name = Path(qs.get("name", [""])[0]).name
            f = Path(CROPS_DIR) / td / name
            if name.endswith(".jpg") and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no image"}, 404); return

        if path == "/identity":
            pid = qs.get("pid", [""])[0]
            _, _, _, idd = ds_dirs()
            f = idd / f"pid_{int(pid):04d}.jpg" if pid.isdigit() else None
            if f and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no identity"}, 404); return

        if path == "/api/dataset":
            # Danh sách toàn bộ identity trong myreid + thống kê tổng
            self._json({"identities": dataset_identities(), "stats": dataset_stats()}); return

        if path == "/api/identity_images":
            pid = qs.get("pid", [""])[0]
            if not pid.isdigit():
                self._json({"error": "bad pid"}, 400); return
            self._json({"pid": int(pid), "images": identity_images(int(pid))}); return

        if path == "/ds_img":
            # Phục vụ 1 ảnh dataset cụ thể (chống path traversal: chỉ lấy basename)
            split = qs.get("split", [""])[0]
            name = Path(qs.get("name", [""])[0]).name
            d = _split_dir(split)
            f = (d / name) if d else None
            if f and name.endswith(".jpg") and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no image"}, 404); return

        # ── Dataset PHỤ (myreid2): xem để đối chiếu trước khi gộp ──
        if path == "/api/dataset2":
            self._json({"present": dataset2_present(),
                        "identities": dataset_identities(DATASET2_DIR),
                        "stats": dataset_stats(DATASET2_DIR)}); return

        if path == "/api/identity_images2":
            pid = qs.get("pid", [""])[0]
            if not pid.isdigit():
                self._json({"error": "bad pid"}, 400); return
            self._json({"pid": int(pid),
                        "images": identity_images(int(pid), DATASET2_DIR)}); return

        if path == "/ds_img2":
            split = qs.get("split", [""])[0]
            name = Path(qs.get("name", [""])[0]).name
            d = _split_dir(split, DATASET2_DIR)
            f = (d / name) if d else None
            if f and name.endswith(".jpg") and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no image"}, 404); return

        if path == "/api/zones":
            self._json({"zones": load_zones()}); return

        if path == "/api/qr_folders":
            self._json({"folders": qr_folders()}); return

        if path == "/qr_img":
            # Phục vụ 1 ảnh trong folder QR (chống path traversal: chỉ lấy basename)
            folder = _safe_folder_name(qs.get("folder", [""])[0])
            name = Path(qs.get("name", [""])[0]).name
            f = (Path(QR_DIR) / folder / name) if folder else None
            if f and name.endswith(".jpg") and f.exists():
                self._bytes(f.read_bytes(), "image/jpeg"); return
            self._json({"error": "no image"}, 404); return

        self._json({"error": "not found"}, 404)

    def _state_payload(self):
        state = detect_state()
        payload = {"state": state, "dataset": dataset_stats(),
                   "committing": COMMIT_PROGRESS["running"]}   # đang gộp nền → UI hiện lại thanh %
        if state == "idle":
            payload["videos"] = scan_videos()
            payload["zones"] = load_zones()
        elif state == "processing":
            payload["progress"] = {**PROGRESS, "log": list(LOG)}
        elif state == "montage_pending":
            pass
        elif state in ("label", "commit"):
            payload["tracks"] = TRACKS
            payload["stats"] = label_stats()
            payload["cameras"] = sorted({t["cam"] for t in TRACKS})
            payload["ds_pids"] = [d["pid"] for d in existing_identities()]  # ID đã có trong dataset
            payload["next_pid"] = next_pid()
            payload["free_pids"] = free_pids()   # số ID bị bỏ trống (đã xóa) → gợi ý tái dùng
            payload["videos"] = scan_videos()   # video mới trong video/ để "Nạp thêm"
        self._json(payload)

    # --- POST ---
    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/start":
            cameras = body.get("cameras", [])
            if not cameras:
                self._json({"error": "Không có camera"}, 400); return
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            # Đặt running=True NGAY (đồng bộ) để /api/state gọi liền sau đó
            # nhận đúng trạng thái "processing" — tránh race khiến nút như bị "đơ".
            PROGRESS.update(running=True, phase="extract", done=False, error="")
            LOG.clear(); LOG.append("Đang khởi động…")
            threading.Thread(target=run_pipeline, args=(cameras,), daemon=True).start()
            self._json({"ok": True}); return

        if path == "/api/resume_montage":
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            def _resume():
                PROGRESS.update(running=True, phase="montage", done=False, error="")
                LOG.clear(); LOG.append("=== MONTAGE (tiếp tục) ===")
                try:
                    write_config("make_montages.py", {
                        "CROPS_DIR": CROPS_DIR, "OUTPUT_DIR": MONTAGE_DIR,
                        "MIN_IMAGES": MONTAGE_MIN_IMAGES})
                    if not run_step("make_montages.py"):
                        raise RuntimeError("Montage lỗi")
                    scan_to_labels()
                    PROGRESS.update(phase="done", done=True)
                except Exception as e:
                    PROGRESS["error"] = str(e); LOG.append(f"[LỖI] {e}")
                finally:
                    PROGRESS["running"] = False
            threading.Thread(target=_resume, daemon=True).start()
            self._json({"ok": True}); return

        if path == "/api/suggest_ids":
            # Auto gợi ý global ID bằng osnet2.onnx (chạy suggest_ids_onnx.py như subprocess)
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            crops = Path(CROPS_DIR)
            if not (crops.exists() and any(crops.glob("cam*/track*"))):
                self._json({"error": "Chưa có crops — chạy crop trước"}, 400); return
            # Đặt running=True NGAY (đồng bộ) để /api/state kế tiếp thấy "processing"
            PROGRESS.update(running=True, phase="suggest", done=False, error="")
            LOG.clear(); LOG.append("=== GỢI Ý ID (osnet2.onnx) ===")
            def _suggest():
                try:
                    if not run_step("suggest_ids_onnx.py"):
                        raise RuntimeError("suggest_ids_onnx.py lỗi")
                    load_labels_from_disk()          # nạp lại labels.csv đã có ID gợi ý
                    PROGRESS.update(phase="done", done=True)
                except Exception as e:
                    PROGRESS["error"] = str(e); LOG.append(f"[LỖI] {e}")
                finally:
                    PROGRESS["running"] = False
            threading.Thread(target=_suggest, daemon=True).start()
            self._json({"ok": True}); return

        # Các thao tác gán nhãn
        td = body.get("track_dir", "")
        rec = BY_DIR.get(td)

        if path == "/api/label":
            if not rec: self._json({"error": "not found"}, 404); return
            with _LOCK: rec["global_pid"] = int(body.get("global_pid", UNSEEN))
            save_csv(); self._json({"ok": True, "stats": label_stats(),
                                    "next_pid": next_pid(), "free_pids": free_pids()}); return

        if path == "/api/skip":
            if not rec: self._json({"error": "not found"}, 404); return
            with _LOCK: rec["global_pid"] = SKIP
            save_csv(); self._json({"ok": True, "stats": label_stats(),
                                    "next_pid": next_pid(), "free_pids": free_pids()}); return

        if path == "/api/delete":
            if not rec: self._json({"error": "not found"}, 404); return
            folder = Path(CROPS_DIR) / td
            if folder.exists(): shutil.rmtree(str(folder))
            if rec["montage"] and Path(rec["montage"]).exists(): Path(rec["montage"]).unlink()
            with _LOCK:
                TRACKS.remove(rec); BY_DIR.pop(td, None)
            save_csv(); self._json({"ok": True, "stats": label_stats()}); return

        if path == "/api/delete_image":
            # Xóa 1 ảnh lẻ trong track (xử lý ca nhảy ID: 1 track lẫn 2 người)
            name = Path(body.get("name", "")).name
            f = Path(CROPS_DIR) / td / name
            if not (rec and name.endswith(".jpg") and f.exists()):
                self._json({"error": "not found"}, 404); return
            f.unlink()
            with _LOCK:
                remain = len(list((Path(CROPS_DIR) / td).glob("*.jpg")))
                rec["num_images"] = remain
            save_csv()
            self._json({"ok": True, "num_images": rec["num_images"]}); return

        if path == "/api/reset_batch":
            # Xóa TẤT CẢ dữ liệu đợt đang làm (crops, montages, labels, video) — về idle.
            # KHÔNG đụng dataset myreid.
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            ok = True
            for d in (Path(CROPS_DIR), Path(MONTAGE_DIR)):
                if d.exists():
                    ok &= _safe_rmtree(d)
            if Path(LABELS_CSV).exists():
                try: Path(LABELS_CSV).unlink()
                except OSError: ok = False
            for f in _mp4_files(Path(VIDEO_DIR)):
                try: f.unlink()
                except OSError: pass
            with _LOCK:
                if ok:
                    _load_rows([])
            self._json({"ok": ok}); return

        # ── Thao tác trên dataset myreid (trình duyệt dataset) ──
        if path == "/api/ds_delete_images":
            pid = body.get("pid")
            images = body.get("images", [])
            if not isinstance(images, list) or not images:
                self._json({"error": "no images"}, 400); return
            deleted = ds_delete_images(images)
            # Nếu identity không còn ảnh nào → xóa luôn avatar mồ côi
            remaining = identity_images(int(pid)) if isinstance(pid, int) else []
            if isinstance(pid, int) and not remaining:
                _, _, _, idd = ds_dirs()
                av = idd / f"pid_{int(pid):04d}.jpg"
                if av.exists():
                    av.unlink()
            self._json({"ok": True, "deleted": deleted,
                        "remaining": len(remaining), "dataset": dataset_stats()}); return

        if path == "/api/ds_delete_identity":
            pid = body.get("pid")
            if not isinstance(pid, int):
                self._json({"error": "bad pid"}, 400); return
            deleted = ds_delete_identity(pid)
            self._json({"ok": True, "deleted": deleted, "dataset": dataset_stats()}); return

        if path == "/api/ds_delete_identities":
            # Xóa NHIỀU identity 1 lượt — ở dataset chính ("main") hoặc phụ ("aux")
            pids = body.get("pids", [])
            src  = body.get("source", "main")
            base = DATASET2_DIR if src == "aux" else None
            if not isinstance(pids, list) or not pids:
                self._json({"error": "no pids"}, 400); return
            total = sum(ds_delete_identity(p, base) for p in pids if isinstance(p, int))
            self._json({"ok": True, "deleted": total, "removed": len(pids),
                        "dataset": dataset_stats(base)}); return

        if path == "/api/ds_rename_identity":
            # Đổi ID 1 identity (đổi số hoặc gộp vào ID đã có) — main/aux
            old, new = body.get("old_pid"), body.get("new_pid")
            src = body.get("source", "main")
            base = DATASET2_DIR if src == "aux" else None
            if not isinstance(old, int) or not isinstance(new, int) or new < 0:
                self._json({"error": "ID không hợp lệ"}, 400); return
            self._json(ds_rename_identity(old, new, base)); return

        if path == "/api/ds_split_images":
            # Tách các ảnh đã chọn ra 1 ID MỚI (sửa ID bị trộn 2 người)
            pid = body.get("pid")
            images = body.get("images", [])
            src = body.get("source", "main")
            base = DATASET2_DIR if src == "aux" else None
            if not isinstance(pid, int) or not isinstance(images, list) or not images:
                self._json({"error": "thiếu pid/images"}, 400); return
            r = ds_split_images(images, base)
            self._json({"ok": True, **r, "dataset": dataset_stats(base)}); return

        if path == "/api/ds_set_avatar":
            pid = body.get("pid")
            ok = isinstance(pid, int) and ds_set_avatar(pid, body.get("split", ""),
                                                         body.get("name", ""))
            self._json({"ok": bool(ok)}, 200 if ok else 400); return

        # ── Folder QR (gom ảnh chọn tay từ dataset) ──
        if path == "/api/qr_create":
            name = qr_create_folder(body.get("name", ""))
            if not name:
                self._json({"error": "tên folder không hợp lệ"}, 400); return
            self._json({"ok": True, "name": name, "folders": qr_folders()}); return

        if path == "/api/qr_add":
            folder = body.get("folder", "")
            images = body.get("images", [])
            if not _safe_folder_name(folder):
                self._json({"error": "thiếu tên folder"}, 400); return
            if not isinstance(images, list) or not images:
                self._json({"error": "no images"}, 400); return
            copied = qr_add_images(folder, images)
            self._json({"ok": True, "copied": copied, "folders": qr_folders()}); return

        if path == "/api/ds2_pick_import":
            # Mở hộp thoại chọn folder trên máy rồi import luôn
            p = _pick_folder()
            if p is None:
                self._json({"ok": False, "fallback": True,
                            "error": "Không mở được hộp thoại trên máy chủ"}); return
            if p == "":
                self._json({"ok": False, "cancelled": True}); return
            self._json(ds2_import(p)); return

        if path == "/api/ds2_import":
            # Import 1 folder vào dataset phụ (trả 200 kèm ok/error để UI hiện thông báo rõ)
            self._json(ds2_import(body.get("path", ""))); return

        if path == "/api/merge_dataset2":
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            if not dataset2_present():
                self._json({"error": f"Không thấy dataset phụ trong {DATASET2_DIR}/"}, 400); return
            result = merge_dataset2()
            self._json({"ok": True, "result": result}); return

        if path == "/api/commit":
            if PROGRESS["running"]:
                self._json({"error": "Đang chạy"}, 409); return
            # Chặn bấm 'Hoàn tất' 2 lần / 2 request song song → không copy lặp
            if not _COMMIT_LOCK.acquire(blocking=False):
                self._json({"error": "Đang gộp dataset, đợi chút"}, 409); return
            # Chạy NỀN để UI poll thanh tiến trình (batch to copy vài phút)
            COMMIT_PROGRESS.update(running=True, phase="Chuẩn bị…", done=0, total=0,
                                   result=None, error="")

            def _do_commit():
                try:
                    COMMIT_PROGRESS["result"] = commit_to_dataset()
                except Exception as e:
                    COMMIT_PROGRESS["error"] = str(e)
                    print("[LỖI gộp dataset]", e)
                finally:
                    COMMIT_PROGRESS["running"] = False
                    _COMMIT_LOCK.release()

            threading.Thread(target=_do_commit, daemon=True).start()
            self._json({"ok": True, "started": True}); return

        if path == "/api/zones":
            save_zones(body.get("zones", {}))
            self._json({"ok": True}); return

        self._json({"error": "not found"}, 404)


# ── Khởi động ─────────────────────────────────────────────────────────────────
def main():
    import os
    os.chdir(HERE)   # mọi đường dẫn tương đối (crops/, dataset/, video/...) bám theo script

    # Nạp trạng thái hiện có (nếu đang dở)
    if Path(LABELS_CSV).exists():
        load_labels_from_disk()

    url = f"http://{HOST}:{PORT}"
    st = detect_state()
    print(f"\nRe-ID Pipeline server: {url}")
    print(f"Trạng thái hiện tại: {st}")
    print("Nhấn Ctrl+C để dừng.\n")

    if OPEN_BROWSER:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
        server.shutdown()


if __name__ == "__main__":
    main()
