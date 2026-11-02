# The age check's reference outputs

`crates/oaiy-media/src/faceage` ports MiVOLO v2 and runs YuNet; its tests compare the port with the authors' own code,
run here on the CPU. Nothing here is used at run time.

What the references were made from (2 November 2026):

- MiVOLO's code: `git clone https://github.com/WildChlamydia/MiVOLO` at `37475e3f8818b5f22448003feec3e64b01bfb188`.
- MiVOLO v2's weights: `https://huggingface.co/iitolstykh/mivolo_v2/resolve/53393526c220e34cdd7b722b36d22b6f9e5f4241/model.safetensors`
  (115,078,528 bytes, sha256 `96efb47051c038ebeec74b73b4253c5fd000433e5afcab7deee0bd8f3fa7bf18`).
- YuNet: `https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx`
  (232,589 bytes, sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`).
- Python 3.11 with torch, numpy, safetensors, opencv-python and onnxruntime; in a virtual environment of its own,
  timm 0.8.13.dev0 (`pip install --no-deps timm==0.8.13.dev0`), the version MiVOLO pins: timm 1.0 has lost a name its
  code imports, and 0.9's VOLO is not the one its weights were trained on.

```sh
python mivolo_ref.py <MiVOLO clone> <model.safetensors> <out>   # mivolo_ref.safetensors, mivolo_pre.safetensors, the test picture
python yunet_ref.py <face_detection_yunet_2023mar.onnx> <out>   # yunet_raw.safetensors, yunet_cv2.txt (after mivolo_ref.py)
python yunet_graph.py <face_detection_yunet_2023mar.onnx>        # the network's operators, for a reader of the port
```

Then, from the repository:

```sh
OAIY_FACEAGE_MIVOLO=<model.safetensors> OAIY_FACEAGE_YUNET=<face_detection_yunet_2023mar.onnx> OAIY_FACEAGE_REF=<out> \
  cargo test --release -p oaiy-media --lib faceage -- --nocapture
```

Without those three set the comparisons say they are skipped and pass. What they found on 2 November 2026: MiVOLO's
head within 1.3e-4 and the age within 0.002 years on a fixed random input, and within 0.04 years through the
preprocessing of the test picture (OpenCV's fixed-point resize puts a pixel a level off here and there); YuNet's
outputs within 2e-6 of onnxruntime's, and the same candidates as OpenCV's `FaceDetectorYN` (boxes within half a
pixel, scores within 0.001). The test picture is a synthetic gradient: no photograph of a person is used.
