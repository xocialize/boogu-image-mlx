"""Full-DiT PT-vs-MLX parity (CPU stream, fp32, tiny input).

MLX output is computed and freed before the torch oracle loads, to keep peak
memory near a single fp32 copy of the ~10B-param transformer.
"""

import gc
import json
import os
import sys

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ORACLE = os.path.expanduser("~/Development/mlxengine-image/boogu-oracle")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from _helpers import make_seeded_input  # noqa: E402

BASE = os.path.expanduser(os.environ.get("BOOGU_BASE",
        "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
TDIR = os.path.join(BASE, "transformer")
CFG = json.load(open(os.path.join(TDIR, "config.json")))

# Shared seeded inputs
LAT = make_seeded_input((1, 16, 16, 16), seed=1)       # [B,C,H,W]
TS = np.array([0.5], dtype=np.float32)
INSTR = make_seeded_input((1, 8, 4096), seed=2)


def run_mlx() -> np.ndarray:
    from boogu_image_mlx.models.transformer import BooguImageTransformer2DModel
    from boogu_image_mlx.utils.weights import read_safetensors_dir, load_named_into_mlx

    model = BooguImageTransformer2DModel.from_config(CFG)
    sd = read_safetensors_dir(TDIR, dtype=mx.float32)
    load_named_into_mlx(model, sd)
    del sd
    out = model(mx.array(LAT), mx.array(TS), mx.array(INSTR))
    mx.eval(out)
    arr = np.array(out)
    del model, out
    gc.collect()
    mx.clear_cache()
    return arr


def run_torch() -> np.ndarray:
    sys.path.insert(0, ORACLE)
    import torch
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel as TDiT

    model = TDiT.from_pretrained(TDIR, torch_dtype=torch.float32).eval()
    theta = 10000
    freqs_cis = model.rope_embedder.get_freqs_cis(CFG["axes_dim_rope"], CFG["axes_lens"], theta)
    with torch.no_grad():
        out = model(
            hidden_states=[torch.from_numpy(LAT[0]).float()],          # [C,H,W]
            timestep=torch.from_numpy(TS).float(),
            instruction_hidden_states=torch.from_numpy(INSTR).float(),
            freqs_cis=freqs_cis,
            instruction_attention_mask=torch.ones(1, INSTR.shape[1], dtype=torch.long),
            ref_image_hidden_states=None,
            return_dict=False,
        )
    o = out[0] if isinstance(out, (list, tuple)) else out
    o = o[0] if isinstance(o, (list, tuple)) else o
    return o.detach().cpu().float().numpy()


def main() -> int:
    mx_out = run_mlx()[0]            # [C,H,W]
    pt_out = run_torch()            # [C,H,W]
    if pt_out.shape != mx_out.shape:
        print(f"SHAPE MISMATCH pt={pt_out.shape} mx={mx_out.shape}")
        return 1
    d = np.abs(pt_out - mx_out)
    thr = 1e-2
    status = "PASS" if d.max() < thr else "FAIL"
    print(f"[DiT full] {status} max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
          f"(thr {thr}) pt_range=({pt_out.min():.3f},{pt_out.max():.3f}) "
          f"mx_range=({mx_out.min():.3f},{mx_out.max():.3f})")
    return 0 if d.max() < thr else 1


if __name__ == "__main__":
    raise SystemExit(main())
