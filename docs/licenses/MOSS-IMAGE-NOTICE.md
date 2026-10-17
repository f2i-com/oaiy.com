# MOSS-Image reference attribution

The native MOSS-Image implementation (`crates/oaiy-media/src/moss_image/`) and its test fixtures follow OpenMOSS's
MOSS-Image pipeline, `pipeline_moss_image.py`, published with the MOSS-Image-Base and MOSS-Image-SFT weights:
https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT and https://github.com/OpenMOSS/MOSS-Image (Apache License 2.0).
The OpenMOSS authors retain their rights in that source. Rust implementation adaptations and integration are by OAIY
contributors. The complete upstream license accompanies this notice in `Apache-2.0-MOSS-IMAGE.txt`.

Its text conditioning runs Qwen3.5-9B's language model as Hugging Face Transformers' Qwen3.5 implementation does (Apache
License 2.0): https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5, Copyright The Qwen
team, Alibaba Group and the HuggingFace Inc. team. Test reference vectors were evaluated independently on the CPU
(`tools/moss_image/`); the native runtime does not invoke the reference Python implementations.

Model weights remain separate user-supplied files and retain their own licenses: MOSS-Image's (Apache 2.0), Qwen3.5-9B's
(Apache 2.0), and those of third-party quantizations (realrebelai's GGUFs: Apache 2.0) and distillations. A
distillation's sampler (`moss_image::distill`) follows the data-space DMD method (arXiv 2610.12102) as such a
distillation's own files describe it; `tools/moss_image/distill_*.py` evaluate it on the CPU and are never invoked by
the native runtime.
