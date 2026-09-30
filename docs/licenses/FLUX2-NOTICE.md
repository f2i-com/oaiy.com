# FLUX.2 Klein reference attribution

The native Klein implementation and test fixtures follow the Black Forest Labs
FLUX.2 reference implementation: https://github.com/black-forest-labs/flux2
(Apache License 2.0). Black Forest Labs authors retain their rights in that source.
Rust implementation adaptations and integration are by OAIY contributors.
The complete upstream license accompanies this notice in `Apache-2.0-FLUX2.txt`.

Qwen3 conditioning follows Hugging Face Transformers' Qwen3 implementation
(Apache License 2.0): https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3
Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
All rights reserved. Test reference vectors were evaluated independently on CPU;
the native runtime does not invoke the reference Python implementations.

Model weights and adapters remain separate user-supplied files and retain their
own licenses. The official FLUX.2 Klein 4B model repository labels its weights
Apache 2.0; this statement does not apply to other FLUX variants or adapters.
