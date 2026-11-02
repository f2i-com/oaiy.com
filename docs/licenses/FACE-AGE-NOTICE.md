# The age check's attribution (MiVOLO v2 and YuNet)

The age model in `crates/oaiy-media/src/faceage/mivolo.rs` follows MiVOLO's model code
(https://github.com/WildChlamydia/MiVOLO at 37475e3f, Copyright 2023 Irina Tolstykh and Maxim Kuprashevich, Apache
License 2.0), which adapts timm's VOLO (https://github.com/huggingface/pytorch-image-models, 0.8.13, Ross Wightman,
Apache License 2.0), and MiVOLO's preprocessing (`class_letterbox`, `prepare_classification_images`). The MiVOLO
authors retain their rights in that source; the complete upstream license accompanies this notice in
`Apache-2.0-MIVOLO.txt`. Its weights, `iitolstykh/mivolo_v2` (Apache License 2.0), are not distributed with OAIY: a
person downloads them from their publisher (the catalog's Age check). MiVOLO's `setup.py` also names CC BY-SA 4.0;
its LICENSE file and the weights' card say Apache 2.0.

The face detector in `crates/oaiy-media/src/faceage/yunet.rs` runs YuNet's published network
(`face_detection_yunet_2023mar.onnx` from https://huggingface.co/opencv/face_detection_yunet, Copyright (c) 2020 Shiqi
Yu, MIT License, in `MIT-YUNET.txt`), read from that file as it is and not distributed with OAIY, and decodes its
outputs as OpenCV's `FaceDetectorYN` and `NMSBoxes` do (OpenCV, Apache License 2.0).

Reference outputs for the tests were made by running the authors' own code and onnxruntime and OpenCV on the CPU, on
synthetic inputs; the native runtime runs none of them.

An age from a face is an estimate. MiVOLO v2 is not evaluated for age assurance at a threshold: in an independent
benchmark (arXiv 2602.07815) a MiVOLO model, used at a plain 18-year line, read 21% of minors as adults. A site
using it should set a margin above its age and keep a stronger check for those near the line.
