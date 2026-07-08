"""
suggest_ids_onnx.py
-------------------
Tự gợi ý global ID cho các tracklet trong crops/ bằng osnet2.onnx (TransReID).

Luồng: extract_crops -> make_montages -> [SCRIPT NÀY] -> review trên web -> build_dataset.py

Cách hoạt động:
  1. Quét crops/cam*/track* → mỗi tracklet.
  2. Embed tracklet bằng osnet2.onnx (trung bình vector ảnh, đã L2-normalize).
  3. Gom cụm bằng Union-Find theo cosine similarity:
       - khác camera: ngưỡng CROSS_THRESH
       - cùng camera : cao hơn (+SAME_CAM_EXTRA) vì ByteTrack đã tách tốt
     Ngưỡng đặt THẬN TRỌNG (cao) → thà tách dư còn hơn gộp nhầm (dễ sửa khi review).
  4. Ghi labels.csv với global_pid = số cụm (ID gợi ý). Track quá ít ảnh -> -1.

Sau khi chạy: mở web (label_server) để review/sửa ID, rồi build_dataset.py xuất myReIDNew.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# ===================== CẤU HÌNH — SỬA Ở ĐÂY =====================
CROPS_DIR      = "crops"
MONTAGE_DIR    = "montages"
LABELS_CSV     = "labels.csv"
MODEL_PATH     = "transreid_reid.onnx"   # gom cụm tốt hơn hẳn model tự train (transreid_myreid nén cosine → chaining)
CROSS_THRESH   = 0.60    # ngưỡng cosine cặp KHÁC camera (hạ 0.70→0.60: gộp thoáng hơn, bớt tách 1 người thành nhiều ID)
SAME_CAM_EXTRA = 0.15    # cùng camera cộng thêm chừng này (→ same-cam 0.75)
MIN_IMAGES     = 3       # tracklet ít hơn N ảnh -> bỏ (global_pid = -1)
MAX_EMBED_IMAGES = 8     # embed tối đa N ảnh/track (rải đều) → nhanh hơn nhiều, mean embedding vẫn ổn
CACHE_FILE     = "reid_emb_cache.npz"   # cache embedding tracklet → đổi ngưỡng khỏi embed lại
DATASET_DIR    = "myReIDNew"   # ID gợi ý sẽ đánh số TIẾP theo dataset này (tránh trùng ID cũ) ★
# =================================================================


def dataset_next_pid(dataset_dir):
    """ID thật lớn nhất trong dataset + 1 → offset cho ID gợi ý (khỏi trùng ID đã có).
    Trả 0 nếu dataset trống/không có."""
    import re as _re
    mx = -1
    base = Path(dataset_dir)
    for sub in ("bounding_box_train", "query", "bounding_box_test"):
        d = base / sub
        if d.exists():
            for f in d.glob("*.jpg"):
                m = _re.match(r"(\d+)_c", f.name)
                if m:
                    p = int(m.group(1))
                    if p < 100000:            # bỏ distractor
                        mx = max(mx, p)
    idd = base / "_identities"
    if idd.exists():
        for f in idd.glob("pid_*.jpg"):
            m = _re.match(r"pid_(\d+)", f.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def scan_tracklets(crops_dir: Path):
    """Trả list (cam, track_id, track_dir_rel, [image_paths]) cho mọi track có ảnh."""
    out = []
    for td in sorted(crops_dir.glob("cam*/track*")):
        imgs = sorted(td.glob("*.jpg"))
        if not imgs:
            continue
        m = re.search(r"cam(\d+)", td.parent.name)
        if not m:
            continue
        cam = int(m.group(1))
        track_id = int(td.name.replace("track", ""))
        rel = str(td.relative_to(crops_dir)).replace("\\", "/")
        out.append((cam, track_id, rel, imgs))
    return out


def run(crops_dir=CROPS_DIR, montage_dir=MONTAGE_DIR, labels_csv=LABELS_CSV,
        model_path=MODEL_PATH, cross_thresh=CROSS_THRESH,
        same_cam_extra=SAME_CAM_EXTRA, min_images=MIN_IMAGES, cache_file=CACHE_FILE,
        dataset_dir=DATASET_DIR):
    crops_dir = Path(crops_dir)
    montage_dir = Path(montage_dir)
    tracks = scan_tracklets(crops_dir)
    if not tracks:
        print(f"[LỖI] Không thấy tracklet trong {crops_dir}/ (chạy extract_crops trước).")
        return
    print(f"Tracklet tìm thấy: {len(tracks)}")

    # --- Nạp cache embedding (nếu có) → đổi ngưỡng khỏi embed lại ---
    cache = {}
    if cache_file and Path(cache_file).exists():
        try:
            z = np.load(cache_file, allow_pickle=True)
            cached_model = str(z["model"]) if "model" in z.files else ""
            if cached_model == str(model_path):          # chỉ dùng lại cache NẾU cùng model
                for td, emb, ni in zip(z["dirs"], z["embs"], z["nimgs"]):
                    cache[str(td)] = (emb, int(ni))
            elif cached_model:
                print(f"[cache] model đổi ({cached_model} → {model_path}) → embed lại từ đầu (không trộn 2 model)")
        except Exception:
            cache = {}

    # --- Embed từng tracklet đủ ảnh (dùng cache nếu track không đổi) ---
    def save_cache_now():
        """Ghi toàn bộ cache hiện có ra đĩa → tắt giữa chừng vẫn giữ được phần đã embed."""
        if not cache_file:
            return
        try:
            dirs = list(cache.keys())
            np.savez(cache_file,
                     dirs=np.array(dirs, dtype=object),
                     embs=np.asarray([cache[d][0] for d in dirs], np.float32),
                     nimgs=np.array([cache[d][1] for d in dirs]),
                     model=np.array(str(model_path)))   # nhớ model đã embed → đổi model thì cache tự bỏ
        except Exception as ex:
            print("[CẢNH BÁO] không lưu cache:", ex, flush=True)

    n_todo = sum(1 for (c, t, r, im) in tracks
                 if len(im) >= min_images and not (cache.get(r) and cache[r][1] == len(im)))
    print(f"Cần embed mới: {n_todo} track (còn lại dùng cache)", flush=True)

    extractor = None
    valid, embs = [], []
    reused = embedded = 0
    for i, (cam, tid, rel, imgs) in enumerate(tracks):
        if len(imgs) < min_images:
            continue
        c = cache.get(rel)
        if c is not None and c[1] == len(imgs):
            e = c[0]; reused += 1
        else:
            if extractor is None:
                from reid_onnx import OnnxReID
                extractor = OnnxReID(model_path)
            # Rải đều lấy tối đa MAX_EMBED_IMAGES ảnh → mean embedding vẫn ổn mà nhanh hơn nhiều
            sub = imgs
            if len(imgs) > MAX_EMBED_IMAGES:
                idx = np.linspace(0, len(imgs) - 1, MAX_EMBED_IMAGES).astype(int)
                sub = [imgs[k] for k in idx]
            e = extractor.tracklet_embedding(sub, side_label=cam - 1)   # SIE camera (0-indexed)
            if e is None:
                continue
            embedded += 1
            cache[rel] = (e, len(imgs))
            if embedded % 100 == 0:
                print(f"  ...đã embed {embedded}/{n_todo} track", flush=True)
            if embedded % 500 == 0:          # lưu cache định kỳ → tắt giữa chừng KHÔNG mất công
                save_cache_now()
        valid.append(i); embs.append(e); cache[rel] = (e, len(imgs))
    print(f"Embedding: {embedded} mới · {reused} dùng lại cache", flush=True)
    if not embs:
        print("[LỖI] Không embed được tracklet nào.")
        return

    save_cache_now()   # lưu lần cuối (đổi ngưỡng lần sau khỏi embed lại)

    E = np.asarray(embs)                       # (M, 3840) đã normalize
    sim = E @ E.T                              # cosine

    # --- Union-Find gom cụm ---
    M = len(valid)
    uf = UnionFind(M)
    for a in range(M):
        cam_a = tracks[valid[a]][0]
        for b in range(a + 1, M):
            cam_b = tracks[valid[b]][0]
            thr = (cross_thresh + same_cam_extra) if cam_a == cam_b else cross_thresh
            if sim[a, b] >= thr:
                uf.union(a, b)

    roots = [uf.find(i) for i in range(M)]
    uniq = sorted(set(roots))
    offset = dataset_next_pid(dataset_dir)     # đánh số TIẾP theo dataset → không trùng ID cũ
    root2gid = {r: offset + g for g, r in enumerate(uniq)}
    gid_of_valid = {valid[i]: root2gid[roots[i]] for i in range(M)}
    if offset:
        print(f"ID gợi ý đánh số từ {offset} (tiếp theo dataset '{dataset_dir}', tránh trùng ID cũ)")

    # --- Ghi labels.csv (schema khớp label_server + build_dataset) ---
    fields = ["cam", "track_id", "num_images", "montage", "global_pid", "track_dir"]
    rows = []
    for i, (cam, tid, rel, imgs) in enumerate(tracks):
        mp = montage_dir / f"cam{cam}_track{tid:04d}.jpg"
        rows.append({
            "cam": cam, "track_id": tid, "num_images": len(imgs),
            "montage": str(mp).replace("\\", "/") if mp.exists() else "",
            "global_pid": gid_of_valid.get(i, -1),      # -1 = track ít ảnh / bỏ qua
            "track_dir": rel,
        })
    rows.sort(key=lambda r: (r["global_pid"], r["cam"], r["track_id"]))
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    n_clusters = len(uniq)
    n_skipped = sum(1 for r in rows if r["global_pid"] == -1)
    # phân bố kích thước cụm để bạn ước lượng nhanh
    size = defaultdict(int)
    for r in rows:
        if r["global_pid"] >= 0:
            size[r["global_pid"]] += 1
    multi = sum(1 for s in size.values() if s >= 2)
    print("\n=== KẾT QUẢ GỢI Ý ===")
    print(f"Tracklet embed         : {M}")
    print(f"Số ID gợi ý (cụm)      : {n_clusters}")
    print(f"  trong đó ≥2 tracklet : {multi}  (thường là cross-camera / cùng người)")
    print(f"Track bị bỏ (<{min_images} ảnh) : {n_skipped}  → global_pid = -1")
    print(f"Ngưỡng: cross={cross_thresh}  same-cam={cross_thresh + same_cam_extra}")
    print(f"Đã ghi: {labels_csv}")
    print("\nBước tiếp: mở web (label_server) review/sửa ID → build_dataset.py (OUTPUT_DIR='myReIDNew').")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=CROPS_DIR)
    ap.add_argument("--montages", default=MONTAGE_DIR)
    ap.add_argument("--labels", default=LABELS_CSV)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--cross", type=float, default=CROSS_THRESH)
    ap.add_argument("--same-extra", type=float, default=SAME_CAM_EXTRA)
    ap.add_argument("--min-images", type=int, default=MIN_IMAGES)
    ap.add_argument("--dataset", default=DATASET_DIR, help="Đánh số ID tiếp theo dataset này")
    a = ap.parse_args()
    run(a.crops, a.montages, a.labels, a.model, a.cross, a.same_extra, a.min_images,
        dataset_dir=a.dataset)
