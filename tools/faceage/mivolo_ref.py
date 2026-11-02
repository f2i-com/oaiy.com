"""Reference outputs of MiVOLO v2 (iitolstykh/mivolo_v2 @ 53393526) for checking a port.

Runs the authors' own model code (the MiVOLO clone at 37475e3f) with the published weights, in
float32 on the CPU, on a fixed random 6x384x384 input, and saves the input and the tensors after
each part of the network, so a port can be matched part by part. Also saves the preprocessing of
one fixed picture with given boxes (the authors' letterbox and normalisation).

usage: python mivolo_ref.py <mivolo-src> <model.safetensors> <out-dir>
"""
import sys
from pathlib import Path

src, weights, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(src))

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from mivolo.model.create_timm_model import create_model
from mivolo.data.misc import prepare_classification_images

torch.manual_seed(0)
model = create_model("mivolo_d1_384", num_classes=3, in_chans=6, pretrained=False)
state = load_file(str(weights))
prefix = "mivolo.model."
stripped = {k[len(prefix):]: v.float() for k, v in state.items() if k.startswith(prefix)}
missing, unexpected = model.load_state_dict(stripped, strict=False)
print("missing", missing, "unexpected", unexpected, "tensors", len(stripped))
model.eval()

x = torch.randn(1, 6, 384, 384, generator=torch.Generator().manual_seed(1234))
saved = {"input": x}
with torch.no_grad():
    pe = model.patch_embed
    f1 = pe.proj1(pe.conv1(x[:, :3]))
    f2 = pe.proj2(pe.conv2(x[:, 3:]))
    saved["embed_face"] = f1
    saved["embed_body"] = f2
    fused = pe.map(torch.cat([f1, f2], dim=1))
    saved["fused"] = fused
    t = fused.permute(0, 2, 3, 1)
    # forward_tokens, stage by stage (timm's VOLO)
    for idx, block in enumerate(model.network):
        if idx == 2:
            t = model.pos_embed + t if model.pos_embed is not None else t
            t = model.pos_drop(t)
        t = block(t)
        saved[f"network.{idx}"] = t
    B, H, W, C = t.shape
    t = t.reshape(B, -1, C)
    saved["tokens"] = t
    t = model.forward_cls(t)
    saved["post"] = t
    t = model.norm(t)
    saved["norm"] = t
    head = model.forward_head(t)
    saved["head"] = head
    whole = model(x)
    print("head", head.tolist(), "whole", whole.tolist(), "same", torch.allclose(head, whole, atol=1e-6))
    age = head[:, 2] * (122 - 0) + 61.0
    print("age", age.tolist())
    saved["age"] = age

save_file({k: v.clone().contiguous() for k, v in saved.items()}, str(out / "mivolo_ref.safetensors"))

# Preprocessing of a fixed picture: a 640x480 gradient with a pattern, BGR as cv2 reads it, a
# face box and the frame as the body with the face cut out, as the camera check will do it.
h, w = 480, 640
yy, xx = np.mgrid[0:h, 0:w]
img = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx + yy) * 7 % 256)], axis=-1).astype(np.uint8)
face_box = (250, 120, 410, 330)  # x1, y1, x2, y2
x1, y1, x2, y2 = face_box
face = img[y1:y2, x1:x2].copy()
body = img.copy()
body[y1:y2, x1:x2] = 0
np.save(out / "pre_image_bgr.npy", img)
# The same picture as RGB bytes, as a decoder hands it to the port.
(out / "pre_image_rgb.raw").write_bytes(np.ascontiguousarray(img[:, :, ::-1]).tobytes())
faces = prepare_classification_images([face], 384, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
bodies = prepare_classification_images([body], 384, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
with torch.no_grad():
    pre_head = model(torch.cat([faces, bodies], dim=1))
print("pre faces", tuple(faces.shape), "bodies", tuple(bodies.shape), "head", pre_head.tolist(), "age", (pre_head[:, 2] * 122 + 61).tolist())
save_file({"faces": faces.contiguous(), "bodies": bodies.contiguous(), "head": pre_head.contiguous()}, str(out / "mivolo_pre.safetensors"))
