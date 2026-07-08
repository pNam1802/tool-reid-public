"""
head_detect.py
--------------
Wrapper module for the head-detection model (head_detect.pt).
Model was fine-tuned on top of yolo26n.pt (Ultralytics YOLO26 Nano).

Public API
----------
detect_heads(frame, conf=0.35, iou=0.45) -> list[tuple[int,int,int,int,float]]
    frame : np.ndarray  BGR image (H x W x 3)
    conf  : float       minimum confidence threshold
    iou   : float       NMS IoU threshold
    returns a list of (x1, y1, x2, y2, confidence) for every detected head
"""

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE / "head_detect.pt"

if not _MODEL_PATH.exists():
    raise FileNotFoundError(
        f"[head_detect] Model weights not found: {_MODEL_PATH}\n"
        "Place head_detect.pt in the same directory as head_detect.py."
    )

# ── Lazy singleton ─────────────────────────────────────────────────────────────
_model: YOLO | None = None
_DEVICE = None   # thiết bị đã chốt sau khi load ("cpu" hoặc 0/1/... cho GPU)


def _resolve_device():
    """Chọn thiết bị: ép qua REID_DEVICE nếu có, không thì GPU khả dụng, cuối cùng CPU."""
    env = os.environ.get("REID_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def _load_model() -> YOLO:
    global _model, _DEVICE
    if _model is None:
        _model = YOLO(str(_MODEL_PATH), task="detect")
        _DEVICE = _resolve_device()
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            _model(dummy, device=_DEVICE, verbose=False)   # warm-up
        except Exception as e:
            # GPU lỗi (vd cuDNN: "unable to find an engine") → tự chuyển sang CPU
            if _DEVICE != "cpu":
                print(f"[head_detect] GPU loi ({type(e).__name__}: {e}). Chuyen sang CPU (cham hon nhung chay duoc).")
                _DEVICE = "cpu"
                _model(dummy, device=_DEVICE, verbose=False)
            else:
                raise
    return _model


# ── Public interface ───────────────────────────────────────────────────────────
def detect_heads(
    frame: np.ndarray,
    conf: float = 0.35,
    iou: float = 0.45,
) -> list[tuple[int, int, int, int, float]]:
    """
    Detect human heads in *frame* and return bounding boxes.

    Parameters
    ----------
    frame : np.ndarray
        BGR image of shape (H, W, 3).
    conf : float
        Minimum confidence to keep a detection (default 0.35).
    iou : float
        IoU threshold used during NMS (default 0.45).

    Returns
    -------
    list of (x1, y1, x2, y2, confidence)
        Pixel-integer coordinates in the original frame space, plus the
        float confidence score for each detected head.
        Returns an empty list when no heads are found.
    """
    global _DEVICE
    model = _load_model()
    try:
        results = model(frame, conf=conf, iou=iou, device=_DEVICE, verbose=False)
    except Exception as e:
        # GPU lỗi giữa chừng (cuDNN "unable to find an engine") → chuyển hẳn sang
        # CPU và chạy lại frame này; các frame sau cũng chạy CPU (chậm nhưng xong).
        if _DEVICE != "cpu":
            print(f"[head_detect] GPU loi khi infer ({type(e).__name__}). Chuyen sang CPU.")
            _DEVICE = "cpu"
            results = model(frame, conf=conf, iou=iou, device="cpu", verbose=False)
        else:
            raise

    boxes: list[tuple[int, int, int, int, float]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()      # (N, 4)  float32
        confs = r.boxes.conf.cpu().numpy()     # (N,)    float32
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(c)))

    return boxes
