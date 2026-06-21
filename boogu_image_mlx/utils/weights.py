"""Weight loading helpers for the Boogu-Image MLX port."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten


def load_diffusers_into_mlx(mx_model, state_dict: dict, *, conv_transpose: bool = True) -> list[str]:
    """Load a diffusers-style state_dict (name -> np.ndarray) into an MLX module.

    Names are preserved 1:1 by the MLX port, so the only transform needed is the
    Conv2d weight layout: PyTorch (O,I,H,W) -> MLX (O,H,W,I). Detected by a 4-D
    `.weight` tensor. Linear/GroupNorm tensors pass through unchanged.

    Returns the list of keys that were loaded (for caller-side coverage checks).
    """
    flat: list[tuple[str, mx.array]] = []
    loaded: list[str] = []
    for k, v in state_dict.items():
        arr = np.ascontiguousarray(v)
        if conv_transpose and k.endswith(".weight") and arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
        flat.append((k, mx.array(arr)))
        loaded.append(k)
    mx_model.update(tree_unflatten(flat))
    mx.eval(mx_model.parameters())
    return loaded


def read_safetensors_np(path: str) -> dict:
    """Read a safetensors file into a {name: np.float32 ndarray} dict."""
    from safetensors import safe_open

    out = {}
    with safe_open(path, "numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).astype(np.float32)
    return out


def read_safetensors_dir(dir_path: str, dtype=None) -> dict:
    """Read all *.safetensors shards in a dir into one {name: mx.array} dict.

    Uses mx.load (handles bf16). Pass dtype=mx.float32 to upcast for parity.
    """
    import glob
    import os

    out = {}
    for p in sorted(glob.glob(os.path.join(dir_path, "*.safetensors"))):
        shard = mx.load(p)
        if dtype is not None:
            shard = {k: v.astype(dtype) for k, v in shard.items()}
        out.update(shard)
    return out


def load_named_into_mlx(mx_model, state_dict: dict) -> tuple[int, list, list]:
    """Load a name-matched state_dict (values mx.array or np.ndarray) into an MLX
    module with no transposes. Returns (n_loaded, missing, extra) for coverage.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    model_keys = {k for k, _ in tree_flatten(mx_model.parameters())}
    flat = []
    for k, v in state_dict.items():
        if k not in model_keys:
            continue
        flat.append((k, v if isinstance(v, mx.array) else mx.array(np.ascontiguousarray(v))))
    mx_model.update(tree_unflatten(flat))
    mx.eval(mx_model.parameters())
    loaded = {k for k, _ in flat}
    return len(loaded), sorted(model_keys - loaded), sorted(set(state_dict) - model_keys)
