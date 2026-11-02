"""Reference outputs of YuNet (opencv/face_detection_yunet @ 3cc26e7f, face_detection_yunet_2023mar.onnx) for a port.

1. onnxruntime on a fixed 640x640 BGR input (the test picture, padded below with black): the twelve raw outputs.
2. OpenCV's own FaceDetectorYN on the 640x480 test picture at low score thresholds: what it returns (x, y, w, h,
   ten landmark coordinates, score), so the port's decoding and non-maximum suppression can be compared.

usage: python yunet_ref.py <model.onnx> <ref-dir>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from safetensors.torch import save_file

model, out = sys.argv[1], Path(sys.argv[2])
bgr = np.load(out / "pre_image_bgr.npy")  # 480x640x3, made by mivolo_ref.py
padded = np.zeros((640, 640, 3), np.uint8)
padded[:480] = bgr
x = np.ascontiguousarray(padded.transpose(2, 0, 1)[None].astype(np.float32))

sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
outs = sess.run(None, {"input": x})
saved = {"input": torch.from_numpy(x)}
for n, o in zip(names, outs):
    saved[n] = torch.from_numpy(np.ascontiguousarray(o))
    print(n, o.shape, float(o.min()), float(o.max()))
save_file(saved, str(out / "yunet_raw.safetensors"))

lines = []
for threshold in (0.0, 0.01, 0.05):
    det = cv2.FaceDetectorYN.create(model, "", (640, 480), threshold, 0.3, 5000)
    _, faces = det.detect(bgr)
    faces = np.zeros((0, 15), np.float32) if faces is None else faces
    print("threshold", threshold, "faces", len(faces), faces[:3, [0, 1, 2, 3, 14]].tolist())
    for f in faces:
        lines.append(" ".join([str(threshold)] + [repr(float(v)) for v in f]))
(out / "yunet_cv2.txt").write_text("\n".join(lines) + "\n")
