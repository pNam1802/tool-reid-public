"""
reid_onnx.py
------------
Trích đặc trưng ReID từ osnet2.onnx (thực chất là model TransReID, output 3840-d).

Interface ONNX:
    input      : [1, 3, 256, 128] float   (ảnh RGB đã chuẩn hóa, CHW)
    view_label : [1] int64                 (Side-Info Embedding, để 0)
    feature    : [1, 3840] float

Dùng:
    r = OnnxReID("osnet2.onnx")
    feats = r.embed_paths(["a.jpg", "b.jpg"])   # list vector đã L2-normalize (hoặc None nếu lỗi ảnh)
    sim = float(feats[0] @ feats[1])            # cosine similarity
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class OnnxReID:
    def __init__(self, model_path="osnet2.onnx", size=(256, 128),
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        # Ưu tiên GPU (CUDA) nếu onnxruntime-gpu có sẵn; không thì CPU
        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(str(model_path), providers=prov)

        ins = self.sess.get_inputs()
        self.inp_name = ins[0].name                       # 'image' / 'input'
        # input phụ (nếu có): 'cam_label' (TransReID SIE) hoặc 'view_label' (osnet2)
        self.side_name = next((i.name for i in ins if i.name != self.inp_name), None)
        self.out_name = self.sess.get_outputs()[0].name   # 'feature'

        self.H, self.W = size                             # 256 x 128
        self.mean = np.array(mean, np.float32).reshape(3, 1, 1)
        self.std = np.array(std, np.float32).reshape(3, 1, 1)
        print("[OnnxReID] providers:", self.sess.get_providers())

    def _preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (self.W, self.H))                 # dsize = (width, height)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)                                # HWC -> CHW
        img = (img - self.mean) / self.std
        return img[None]                                            # (1, 3, H, W)

    def embed_image(self, img_bgr, side_label=0):
        """1 ảnh BGR -> vector đã L2-normalize (3840,).
        side_label: giá trị cho input phụ (cam_label 0-indexed cho TransReID, hoặc view_label)."""
        feed = {self.inp_name: self._preprocess(img_bgr)}
        if self.side_name:
            feed[self.side_name] = np.array([int(side_label)], np.int64)
        f = self.sess.run([self.out_name], feed)[0][0]             # (3840,)
        n = np.linalg.norm(f)
        return f / n if n > 1e-8 else f

    def embed_paths(self, paths, side_label=0):
        """List đường dẫn -> list vector (None nếu ảnh hỏng)."""
        out = []
        for p in paths:
            img = cv2.imread(str(p))
            out.append(None if img is None else self.embed_image(img, side_label))
        return out

    def tracklet_embedding(self, paths, side_label=0):
        """Embedding đại diện 1 tracklet = trung bình các vector ảnh (đã L2-normalize).
        side_label: camera 0-indexed của tracklet (cam-1) cho SIE."""
        feats = [f for f in self.embed_paths(paths, side_label) if f is not None]
        if not feats:
            return None
        m = np.mean(feats, axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 1e-8 else m
