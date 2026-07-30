"""Make Lance run on the OAIY Companion's setup (weights off-repo, no flash-attn).

Two idempotent, marker-guarded source patches, applied to the fetched Lance repo
at install time (and re-runnable):

1) config/config_factory.get_model_path -> honor LANCE_MODEL_BASE_DIR.
   Lance has TWO get_model_path helpers. common.gradio_utils.helpers already
   honors the env (used for the main LLM checkpoint), but config.config_factory
   reads path_default.yaml whose values are RELATIVE "downloads/..." (the ViT and
   the Wan VAE use this one). With weights off-repo (LANCE_MODEL_BASE_DIR=
   E:\models\lance) the LLM loads but the ViT/VAE try a relative "downloads/..."
   dir that doesn't exist -> a 404 to HuggingFace. We rewrite a leading
   "downloads"/"downloads/..." to sit under LANCE_MODEL_BASE_DIR.

2) modeling/vit/qwen2_5_vl_vit.Qwen2_5_VisionTransformerPretrainedModel ->
   pin attn_implementation="sdpa". The vision tower otherwise requests
   flash_attention_2, and transformers validates it via
   importlib.metadata.version("flash_attn") -> PackageNotFoundError with our
   metadata-less SDPA shim. The ViT has a real SDPA branch
   (apply_rotary_pos_emb_vision + F.scaled_dot_product_attention) that is
   numerically identical to its flash branch, so pinning SDPA is free.

Usage (run by install-lance.bat after fetching the repo):
    python patch_lance_paths.py <path-to-Lance-repo-root>
Back-compat: if given a path ending in config_factory.py, only patch #1 runs.
"""
import os
import sys

# ---- patch #1: config_factory.get_model_path ----------------------------------
PATHS_MARKER = "OAIY_PATH_PATCH"
PATHS_NEEDLE = '    return str(value) if value is not None else ""'
PATHS_REPLACEMENT = '''    value = str(value) if value is not None else ""
    # OAIY_PATH_PATCH: the bundled path_default.yaml uses a relative "downloads/..."
    # base; honor LANCE_MODEL_BASE_DIR so the ViT/VAE resolve next to off-repo weights.
    import os as _os
    _base = _os.getenv("LANCE_MODEL_BASE_DIR")
    if _base and value:
        _v = value.replace("\\\\", "/")
        if _v == "downloads" or _v.startswith("downloads/"):
            from pathlib import Path as _P
            _rel = _v[len("downloads/"):] if _v.startswith("downloads/") else ""
            value = str(_P(_base) / _rel) if _rel else str(_P(_base))
    return value'''

# ---- patch #2: ViT attn_implementation -> sdpa --------------------------------
VIT_MARKER = "OAIY_VIT_SDPA"
VIT_NEEDLE = (
    "    def __init__(self, config, *inputs, **kwargs) -> None:\n"
    "        super().__init__(config, *inputs, **kwargs)\n"
)
VIT_REPLACEMENT = (
    "    def __init__(self, config, *inputs, **kwargs) -> None:\n"
    "        # OAIY_VIT_SDPA: pin the vision tower to SDPA so it never hits transformers'\n"
    "        # flash-attn validation (no Windows/Blackwell flash_attn wheel; the LLM trunk\n"
    "        # uses our SDPA flash_attn shim). The ViT's sdpa branch is the same math.\n"
    "        try:\n"
    "            config._attn_implementation = \"sdpa\"\n"
    "        except Exception:\n"
    "            pass\n"
    "        super().__init__(config, *inputs, **kwargs)\n"
)


def _patch_file(path: str, marker: str, needle: str, replacement: str, label: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError as e:
        print(f"[patch_lance_paths] {label}: could not read {path}: {e}")
        return False
    if marker in src:
        print(f"[patch_lance_paths] {label}: already patched")
        return True
    if needle not in src:
        print(f"[patch_lance_paths] {label}: WARNING expected anchor not found; "
              "upstream may have changed. Left untouched.")
        return True  # non-fatal
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(needle, replacement, 1))
    print(f"[patch_lance_paths] {label}: patched")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("[patch_lance_paths] usage: patch_lance_paths.py <Lance-repo-root | config_factory.py>")
        return 1
    arg = sys.argv[1]

    # Back-compat: a direct config_factory.py path -> just patch #1.
    if arg.replace("\\", "/").endswith("config_factory.py"):
        ok = _patch_file(arg, PATHS_MARKER, PATHS_NEEDLE, PATHS_REPLACEMENT, "get_model_path")
        return 0 if ok else 1

    repo = arg
    cf = os.path.join(repo, "config", "config_factory.py")
    vit = os.path.join(repo, "modeling", "vit", "qwen2_5_vl_vit.py")
    ok1 = _patch_file(cf, PATHS_MARKER, PATHS_NEEDLE, PATHS_REPLACEMENT, "get_model_path")
    ok2 = _patch_file(vit, VIT_MARKER, VIT_NEEDLE, VIT_REPLACEMENT, "vit-sdpa")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
