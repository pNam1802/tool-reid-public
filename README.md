# Re-ID Data Pipeline

Công cụ **thu thập & gán nhãn** dữ liệu Person Re-Identification (Re-ID) từ hệ
thống nhiều camera, xây dựng dataset chuẩn **Market-1501** để train model Re-ID
(TransReID, OSNet…).
Toàn bộ chạy qua **một web app duy nhất** — `label_server.py` — tự dẫn bạn theo
từng bước, giữ trạng thái, lưu kết quả tự động.

```
   Bỏ video vào video/  ──►  python label_server.py  ──►  mở http://127.0.0.1:8000
                                       │
   ┌───────────────────────────────────┴──────────────────────────────────────┐
   │ (tuỳ chọn) Vẽ Zone  → chỉ lấy người trong vùng quan tâm mỗi camera          │
   │ 1. Crop             → cắt ảnh upbody theo track (ByteTrack)  → crops/   TẠM  │
   │ 2. Montage          → ảnh lưới mỗi track để gán nhãn         → montages/ TẠM │
   │ 3. Gán ID           → gán ID thủ công bằng bàn phím          → labels.csv TẠM│
   │ 4. Gộp vào dataset  → đưa ảnh đã gán vào Market-1501         → myreid/  GIỮ  │
   └───────────────────────────────────┬──────────────────────────────────────┘
                                        │  Gộp xong: XÓA sạch crops/ montages/
                                        ▼  labels.csv + video đã xử lý.
                                  myreid/  ──►  train model Re-ID
```

**Thu thập liên tục:** mỗi lần có video mới, lặp lại các bước; dataset tích lũy
lớn dần. Người đã có được hiển thị lại để **tái dùng ID cũ** → giữ nhất quán
identity qua các đợt và các camera.

---

## Tính năng

- 🎬 **Pipeline web trọn gói**: crop → montage → gán ID → gộp dataset, không cần
  dòng lệnh, không mất state khi đóng/mở lại.
- 📌 **Vẽ Zone (ROI)** cho từng camera trên **frame thật của video** — chỉ giữ
  người có chân nằm trong vùng quan tâm (giảm nhiễu, dataset sạch hơn).
- ⌨️ **Gán ID bằng bàn phím**, gợi ý số ID kế tiếp, tái dùng ID người cũ.
- 👥 **Gallery "Người đã có"**: xem/đối chiếu mọi identity đang có khi gán nhãn.
- 🗂 **Trình duyệt Dataset**: xem từng ID, mọi ảnh theo split/camera, **kéo
  chọn nhiều ảnh để xóa**, đặt ảnh đại diện, xóa hẳn người.
- 🤖 **Gợi ý ID tự động (ReID)**: gom cụm các track cùng người bằng model ReID
  (TransReID/OSNet ONNX) rồi điền sẵn ID để bạn chỉ việc soát/sửa — *cần bạn tự
  thêm file model*, xem [§1 Cài đặt](#1-cài-đặt).
- 🧹 **Chống camera-bias**: người chỉ xuất hiện ở 1 camera được đưa vào gallery
  làm *distractor* (không thành identity train) — đánh giá sát thực tế hơn.

---

## 1. Cài đặt

```bash
pip install ultralytics supervision opencv-python numpy onnxruntime
# GPU: dùng onnxruntime-gpu thay cho onnxruntime để bước Crop + Gợi ý ID nhanh hơn
```

- **`label_server.py`** (web + điều phối) chỉ dùng **thư viện chuẩn Python** —
  không cần fastapi/streamlit. Các thư viện trên là để `extract_crops.py`
  (YOLO + ByteTrack + OpenCV) và `suggest_ids_onnx.py` (ReID ONNX) chạy.
- **Python 3.10+**. Có **GPU** thì Crop + Gợi ý ID nhanh hơn nhiều (không bắt buộc).

### Model cần có

| Model | Dùng cho | Trong repo? |
|---|---|---|
| **`head_detect.pt`** (~15MB) | detect đầu người ở bước **Crop** | ✅ **đã kèm** — Crop chạy ngay |
| **`transreid_reid.onnx`** (ReID) | tính năng **🤖 Gợi ý ID** | ❌ **bạn tự thêm** (quá lớn cho GitHub) |

> 🤖 **File model ReID KHÔNG có trong repo** (đã `.gitignore` mọi `*.onnx/*.pth`).
> Bạn tự đặt model ReID của mình tên **`transreid_reid.onnx`** ở thư mục gốc,
> hoặc sửa `MODEL_PATH` trong `suggest_ids_onnx.py`. Model cần input
> `[1,3,256,128]` + 1 input phụ (`cam_label`/`view_label`) và output feature
> (chuẩn export TransReID/OSNet — xem `reid_onnx.py`). **Không có model vẫn dùng
> được bình thường**, chỉ là gán ID tay thay vì bấm nút Gợi ý.

> 📦 **Dataset & video KHÔNG nằm trong repo** (đã `.gitignore`). Mỗi người tự bỏ
> video vào `video/`; thư mục `myreid/` sẽ tự sinh và lớn dần trên máy bạn.

---

## 2. Chạy nhanh

```bash
python label_server.py
```

Tự mở `http://127.0.0.1:8000`. Web tự nhận biết đang ở bước nào:

| Trạng thái | Ý nghĩa |
|---|---|
| **idle** (sạch) | cho bỏ video vào `video/`, (tuỳ chọn) vẽ zone, rồi bấm *Bắt đầu Crop + Montage* |
| **processing** | đang crop/montage — xem log trực tiếp |
| **montage_pending** | có crop nhưng chưa montage (server tắt giữa chừng) → bấm làm nốt |
| **label** | còn track chưa gán → màn gán ID |
| **commit** | đã gán xong → *Xem lại theo ID* rồi *Gộp vào dataset* |

---

## 3. Quy trình chi tiết

### Bước 0 (tuỳ chọn) — Vẽ Zone cho từng camera

Ở màn **idle**, mỗi video có nút **📌 Vẽ Zone**:

1. Bấm → web lấy **frame thật của video** ra (kéo slider *Vị trí frame* để đổi
   frame nền; 0% = frame đầu).
2. **Click các điểm** để khoanh vùng quan tâm (đa giác tự do), click lại điểm
   đầu (≤20px) để đóng polygon. Cần ≥ 3 điểm.
3. **Lưu zone**.

Mỗi camera **một zone**, lưu vĩnh viễn trong `zones.json`. Khi Crop, chỉ người có
**chân (bottom-center)** nằm trong zone mới được giữ lại.

### Bước 1–2 — Crop + Montage (tự động)

Bấm *Bắt đầu Crop + Montage*. Server lần lượt:

- **Crop** (`extract_crops.py`): đọc video → detect đầu người mỗi frame → mở rộng
  thành vùng **upbody** → theo dõi qua các frame bằng **ByteTrack** → lưu crop
  vào `crops/cam{C}/track{NNNN}/`. Mỗi *track* = một người được theo dõi liên tục.
- **Montage** (`make_montages.py`): mỗi track ghép thành **1 ảnh lưới** để gán
  nhãn nhanh bằng mắt (track quá ít ảnh sẽ bị bỏ qua, mặc định < 3 ảnh).

### Bước 3 — Gán ID (thủ công — quan trọng nhất)

Đây là bước duy nhất cần con người: xác nhận **track nào là người nào**.

**Phím tắt (gán không cần chuột):**

| Phím | Hành động |
|---|---|
| `0`–`9` | gõ số ID |
| `Enter` | gán ID vừa gõ + tự sang track kế |
| `Space` | gán **cùng ID với track trước** + sang kế |
| `→` / `←` | sang track kế / trước (không gán) |
| `S` | bỏ qua track này (nhiễu/mờ) — không đưa vào dataset |
| `Delete` | xóa hẳn track (ảnh gốc + montage + dòng nhãn, có xác nhận) |

**Quy tắc ID duy nhất:**

> **Cùng một người ngoài đời = cùng một ID** — bất kể khác camera, khác track
> hay khác đợt. Khác người = khác ID. Web gợi ý sẵn số ID mới kế tiếp.

**Giao diện:**
- **Giữa** — montage track đang xét + thông tin cam/track/số ảnh + dải ảnh trong
  track (bấm `×` để xóa lẻ ảnh dính người khác khi track bị nhảy ID).
- **Cột phải "ID đã gán"** — 1 ảnh đại diện mỗi ID; bấm để xem to / tái dùng.
- **Nút "👥 Người đã có"** — gallery toàn bộ identity đang có trong dataset, tìm
  theo ID; bấm 1 người để gán lại ID cũ cho track hiện tại.
- **"Xem lại theo ID"** — gom track theo ID để soát lại trước khi gộp.
- **Nút "🤖 Gợi ý ID"** — chạy `suggest_ids_onnx.py`: embed mọi track bằng model
  ReID rồi gom cụm cùng người (Union-Find theo cosine) và điền sẵn ID để bạn chỉ
  soát/sửa. *Ngưỡng chặt để thà tách dư còn hơn gộp nhầm.* Cần file model ReID
  (xem §1); không có model thì bỏ qua nút này, gán tay như thường.

**Quy trình gán hiệu quả:** label hết **Cam 1** trước (mỗi người 1 ID mới) →
sang **Cam 2/3**: đối chiếu "Người đã có" / "ID đã gán", người cũ gõ đúng ID cũ,
người mới dùng số gợi ý.

**Trường hợp đặc biệt:**
- *Một người bị nhảy ID thành nhiều track* → gán **cùng ID** cho tất cả (đừng
  xóa — càng nhiều ảnh càng tốt).
- *Crop dính 2 người* → Bỏ qua (`S`) hoặc Xóa (`Delete`), hoặc xóa lẻ ảnh xấu.

Nhãn lưu vào `labels.csv`, **tự động lưu** sau mỗi thao tác.

### Bước 4 — Gộp vào dataset

Bấm *Hoàn tất & gộp vào dataset*. Server đọc nhãn, copy ảnh thành Market-1501:

- **Tên file:** `{pid:04d}_c{cam}s1_{frame:06d}_{k:02d}.jpg`
- **Chia theo identity:**
  - Người mới **≥2 camera** → thành identity; ~30% (`TEST_RATIO`) vào **test**,
    còn lại **train**. Người trong test không xuất hiện trong train.
  - Người mới **chỉ 1 camera** → đưa vào **gallery làm distractor** (chống
    camera-bias), không thành identity train. (Đổi `SINGLE_CAM_MODE="drop"` để
    bỏ hẳn.)
  - Người **đã có** trong dataset → bổ sung thêm ảnh vào đúng split cũ.
- Mỗi identity lưu 1 **ảnh đại diện** ở `myreid/_identities/pid_XXXX.jpg`.

Gộp xong: **xóa sạch** `crops/`, `montages/`, `labels.csv` và video đã xử lý;
chỉ giữ `myreid/` lớn dần. Màn idle hiện thống kê "sức khỏe dataset" (mỗi
identity ở mấy camera, tỉ lệ ≥2 camera).

### Quản lý dataset — nút 🗂 **Dataset** (mọi lúc)

- Lưới tất cả ID (avatar + số ảnh + số camera), tìm theo ID, lọc *người thật /
  1 camera / distractor*.
- Bấm 1 ID → xem **mọi ảnh** theo split (train/query/gallery) & camera.
- **Kéo chuột chọn vùng** nhiều ảnh → xóa hàng loạt; đặt ảnh đại diện; xóa hẳn
  cả người.

---

## 4. Cấu trúc thư mục

```
ReID/
├── label_server.py       # ⭐ web app điều phối toàn bộ pipeline
├── label_ui.html         # giao diện web
├── extract_crops.py      # Bước 1: cắt crop theo track (ByteTrack)
├── make_montages.py      # Bước 2: ghép ảnh lưới mỗi track
├── extract_frame.py      # lấy 1 frame video cho trình vẽ zone
├── head_detect.py / .pt  # model detect đầu người (KHÔNG SỬA)
├── scale_box.py          # mở rộng box đầu → vùng upbody (KHÔNG SỬA)
├── suggest_ids_onnx.py   # 🤖 Gợi ý ID: gom cụm track cùng người bằng ReID
├── reid_onnx.py          # wrapper trích feature ReID từ file .onnx
├── transreid_reid.onnx   # ⚠️ MODEL ReID — bạn TỰ THÊM (không có trong repo)
│
├── video/                # 📂 ĐẶT VIDEO VÀO ĐÂY  (không commit)
├── zones.json            # zone (ROI) theo camera         (tự sinh)
├── crops/                # ảnh crop theo track             (TẠM, tự sinh)
│   └── cam1/track0001/f000008.jpg ...
├── montages/             # ảnh lưới để gán nhãn            (TẠM, tự sinh)
├── labels.csv            # nhãn đợt hiện tại               (TẠM, tự sinh)
│
└── myreid/               # 🎯 DATASET Market-1501          (GIỮ, không commit)
    ├── bounding_box_train/
    ├── query/
    ├── bounding_box_test/   # gồm cả distractor (pid ≥ 100000)
    └── _identities/         # ảnh đại diện mỗi ID (pid_XXXX.jpg)
```

> Các thư mục dữ liệu (`video/`, `crops/`, `montages/`, `myreid/`) và `labels.csv`
> đã được `.gitignore` — repo **chỉ chứa code** + `head_detect.pt`. Model ReID
> (`*.onnx/*.pth`), dataset và mọi ảnh người **không** nằm trong repo (PII).

---

## 5. Cấu hình

Mở **`label_server.py`**, sửa block CONFIG ở đầu file:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SAVE_EVERY` | 8 | lưu 1 crop mỗi N frame (tracker vẫn chạy mọi frame) |
| `MIN_HEIGHT` | 80 | bỏ crop thấp hơn N px (người quá xa) |
| `MAX_PER_TRACK` | 25 | tối đa số ảnh mỗi track |
| `MIN_GAP` | 24 | cách tối thiểu N frame giữa 2 ảnh cùng track (chống trùng) |
| `MONTAGE_MIN_IMAGES` | 3 | track ít hơn N ảnh sẽ không tạo montage |
| `TEST_RATIO` | 0.3 | tỉ lệ identity (≥2 cam) đưa vào test khi gộp |
| `SINGLE_CAM_MODE` | `"distractor"` | người 1 camera: `distractor` (vào gallery) hoặc `drop` (bỏ) |
| `HOST` / `PORT` | 127.0.0.1 / 8000 | địa chỉ web server |

**Tính năng 🤖 Gợi ý ID** cấu hình trong `suggest_ids_onnx.py`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_PATH` | `transreid_reid.onnx` | đường dẫn model ReID của bạn |
| `CROSS_THRESH` | 0.60 | ngưỡng cosine gom 2 track **khác** camera (thấp → gom thoáng hơn) |
| `SAME_CAM_EXTRA` | 0.15 | cùng camera cộng thêm ngần này (→ 0.75) |

---

## 6. Thu thập nhiều đợt (tích lũy)

```
Đợt 1: video → crop → montage → gán ID → gộp
Đợt 2: video MỚI → crop → montage → gán ID → gộp   (myreid/ to hơn đợt 1)
```

- Mỗi lần Crop **đánh số track nối tiếp** trong `crops/cam{C}/` — không đè dữ
  liệu cũ.
- Khi gán ID đợt mới: người **đã có** → gõ đúng ID cũ (đối chiếu "👥 Người đã
  có"); người mới → ID mới gợi ý. ID mới **không tái dùng** số đã có.
- Mỗi lần Gộp chỉ **thêm** ảnh mới vào `myreid/`, không build lại từ đầu.

---

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Model weights not found: head_detect.pt` | đặt `head_detect.pt` cạnh `head_detect.py` |
| `FutureWarning: ByteTrack ... deprecated` | chỉ là cảnh báo của supervision, bỏ qua |
| Bước Crop chạy rất lâu | bình thường — detect mỗi frame; máy không GPU sẽ chậm |
| Trình vẽ zone không hiện frame | đảm bảo có `extract_frame.py` + opencv; hard refresh trình duyệt |
| Giao diện không cập nhật sau khi sửa | hard refresh (`Ctrl+Shift+R`) — kiểm badge phiên bản cạnh logo |
| Crop dính 2 người trong 1 ảnh | Bỏ qua / Xóa track, hoặc xóa lẻ ảnh ở bước gán ID |
| `tập test rỗng` | chưa có ai xuất hiện ở ≥2 camera → gán cùng người qua nhiều camera |

---

## 8. Train model Re-ID

Sau khi gộp, trỏ config model (TransReID/OSNet…) vào thư mục `myreid/` — cấu
trúc giống hệt Market-1501 (`bounding_box_train`, `query`, `bounding_box_test`).

---

## Script phụ (không bắt buộc)

- `build_dataset.py` — dựng dataset Market-1501 từ `labels.csv` bằng **dòng lệnh**
  (bản độc lập của bước "Gộp vào dataset"; web đã làm sẵn việc này).
- `pipeline.py` (dashboard Streamlit cũ), `label_manual.py` (gán nhãn Streamlit
  cũ), `embed_and_suggest.py` (gợi ý gộp track đời cũ) — công cụ **đời cũ**,
  không thuộc luồng chính `label_server.py`, giữ lại để tham khảo.
