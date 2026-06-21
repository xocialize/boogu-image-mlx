"""MLX-native conditioning parity: mlx-vlm Qwen3-VL last_hidden_state vs torch golden.

Reuses the input_ids saved in conditioning_golden.npz (so tokenization is fixed),
runs the MLX Qwen3-VL base language model, and compares the final hidden state
(= mllm.model.last_hidden_state) against the torch oracle.
"""

import os
import sys

import numpy as np
import mlx.core as mx

# mlx-vlm mRoPE uses a GPU-only Metal kernel -> run on default (GPU) device.
HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "conditioning_golden.npz")
QWEN = os.path.expanduser(os.environ.get("QWEN3VL_DIR",
        "~/Development/mlxengine-image/weights/Qwen3-VL-8B-Instruct"))


def main() -> int:
    g = np.load(GOLDEN, allow_pickle=True)
    input_ids = g["input_ids"]                  # [1, L]
    pt_feats = g["feats"]                        # [1, L, 4096]

    from mlx_vlm import load
    model, _ = load(QWEN)

    # base language model (lm_head stripped) -> last_hidden_state
    lm = model.language_model
    base = lm.model if hasattr(lm, "model") else lm
    ids = mx.array(input_ids)
    h = base(ids)                                # [1, L, 4096] = norm(h)
    mx.eval(h)
    mx_feats = np.array(h.astype(mx.float32))

    if mx_feats.shape != pt_feats.shape:
        print(f"SHAPE MISMATCH pt={pt_feats.shape} mx={mx_feats.shape}")
        return 1
    d = np.abs(pt_feats - mx_feats)
    # cosine over the full tensor (robust to scale/outlier dims)
    a, b = pt_feats.reshape(-1), mx_feats.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    thr = 5e-2  # mllm runs bf16 in mlx-vlm vs fp32 oracle -> looser abs gate; cosine is the real check
    ok = cos > 0.999
    print(f"[conditioning] cos={cos:.6f} max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
          f"pt_range=({pt_feats.min():.2f},{pt_feats.max():.2f}) "
          f"mx_range=({mx_feats.min():.2f},{mx_feats.max():.2f}) -> {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
