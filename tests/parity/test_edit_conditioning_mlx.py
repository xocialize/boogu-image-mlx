"""MLX TI2I conditioning parity: mlx-vlm Qwen3-VL last_hidden_state (image+text)
vs torch golden. Bypasses lm_head (identity) to surface the hidden state, feeding
the golden's exact processor tensors (GPU — mRoPE/vision are GPU kernels)."""
import os
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "edit_conditioning_golden.npz")
QWEN = os.path.expanduser(os.environ.get("QWEN3VL_DIR",
        "~/Development/mlxengine-image/weights/Qwen3-VL-8B-Instruct"))


class _Identity(nn.Module):
    def __call__(self, x):
        return x


def main() -> int:
    g = np.load(GOLDEN, allow_pickle=True)
    from mlx_vlm import load
    model, _ = load(QWEN)

    orig = model.language_model.lm_head
    model.language_model.lm_head = _Identity()
    h = model(
        mx.array(g["input_ids"]),
        pixel_values=mx.array(g["pixel_values"]),
        image_grid_thw=mx.array(g["image_grid_thw"]),
        mask=None,
    )
    model.language_model.lm_head = orig
    h = h.logits if hasattr(h, "logits") else h
    mx.eval(h)
    mx_feats = np.array(h.astype(mx.float32))
    pt_feats = g["feats"]

    if mx_feats.shape != pt_feats.shape:
        print(f"SHAPE MISMATCH pt={pt_feats.shape} mx={mx_feats.shape}")
        return 1
    a, b = pt_feats.reshape(-1), mx_feats.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    d = np.abs(pt_feats - mx_feats)
    ok = cos > 0.999
    print(f"[edit conditioning] cos={cos:.6f} max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
          f"-> {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
