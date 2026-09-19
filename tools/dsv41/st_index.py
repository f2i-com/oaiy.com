"""Raw safetensors index over a sharded checkpoint directory.

Reads only the JSON headers, so it needs no `safetensors` package support for
the checkpoint's F8_E8M0 dtype, and hands out tensors either as raw bytes
(one positioned read) or as numpy memmaps (for row lookups into the 101 GB
Engram tables). This is also the exact layout the Rust `SafetensorsStore`
reads, so the two must agree on every offset.
"""

import json
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

# safetensors dtype tag -> (torch dtype used to view the raw bytes, element size)
DTYPES = {
    "F32": (torch.float32, 4),
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E8M0": (torch.float8_e8m0fnu, 1),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: str  # absolute path of the shard
    dtype: str  # safetensors dtype tag
    shape: tuple
    start: int  # absolute byte offset of the data inside the shard
    nbytes: int


class StIndex:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tensors: dict[str, TensorInfo] = {}
        for f in sorted(os.listdir(model_dir)):
            if not f.endswith(".safetensors"):
                continue
            path = os.path.join(model_dir, f)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            hdr.pop("__metadata__", None)
            base = 8 + n
            for k, v in hdr.items():
                a, b = v["data_offsets"]
                self.tensors[k] = TensorInfo(k, path, v["dtype"], tuple(v["shape"]), base + a, b - a)

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def info(self, name: str) -> TensorInfo:
        return self.tensors[name]

    def read_bytes(self, name: str) -> bytes:
        t = self.tensors[name]
        with open(t.file, "rb") as fh:
            fh.seek(t.start)
            data = fh.read(t.nbytes)
        if len(data) != t.nbytes:
            raise IOError(f"short read for {name}: {len(data)} of {t.nbytes}")
        return data

    def get(self, name: str, device="cpu") -> torch.Tensor:
        """The tensor in its stored dtype (float8 kinds included), on `device`."""
        t = self.tensors[name]
        dtype, _ = DTYPES[t.dtype]
        # numpy -> torch keeps this on the CPU even when a default CUDA device is set
        raw = torch.from_numpy(np.frombuffer(self.read_bytes(name), dtype=np.uint8).copy())
        return raw.view(dtype).reshape(t.shape).to(device)

    def memmap_rows(self, name: str) -> np.ndarray:
        """uint8 [rows, row_bytes] memmap; row lookups touch only the pages they need."""
        t = self.tensors[name]
        _, esize = DTYPES[t.dtype]
        rows = t.shape[0]
        row_bytes = t.nbytes // rows
        assert row_bytes == int(np.prod(t.shape[1:])) * esize
        return np.memmap(t.file, dtype=np.uint8, mode="r", offset=t.start, shape=(rows, row_bytes))
