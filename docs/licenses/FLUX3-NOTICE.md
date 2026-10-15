# FLUX 3 Image Klein reference attribution

The native FLUX 3 Image Klein implementation (`crates/oaiy-media/src/flux3_wgpu.rs`, `crates/oaiy-media/src/flux3/`)
and its test fixtures follow the Black Forest Labs FLUX 3 Image Klein reference implementation:
https://github.com/black-forest-labs/flux-3-image-klein (Apache License 2.0, Copyright 2026 Black Forest Labs, Inc.).
Black Forest Labs authors retain their rights in that source. Rust implementation adaptations and integration are by
OAIY contributors. The complete upstream license accompanies this notice in `Apache-2.0-FLUX3.txt`.

Its text conditioning runs Qwen3-VL-4B-Instruct's language model as Hugging Face Transformers' Qwen3-VL implementation
does (Apache License 2.0): https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl
Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved. Test reference vectors
were evaluated independently on the CPU (`tools/flux3/`); the native runtime does not invoke the reference Python
implementations.

Model weights remain separate user-supplied files and retain their own licenses. FLUX 3 Image Klein's weights are
released under the FLUX Kommunity License v1.0 (non-commercial use, with a commercial-output exception for qualifying
users; see its terms at https://huggingface.co/black-forest-labs/flux-3-image-klein), which OAIY neither distributes
nor relicenses. The Qwen3-VL-4B-Instruct text encoder is Apache 2.0.
