"""DiT load + forward smoke (CPU stream): weight coverage, shape, finiteness."""

import json
import os
import sys

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from boogu_image_mlx.models.transformer import BooguImageTransformer2DModel  # noqa: E402
from boogu_image_mlx.utils.weights import read_safetensors_dir, load_named_into_mlx  # noqa: E402
from _helpers import make_seeded_input  # noqa: E402


def main() -> int:
    base = os.path.expanduser(os.environ.get("BOOGU_BASE",
            "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
    tdir = os.path.join(base, "transformer")
    cfg = json.load(open(os.path.join(tdir, "config.json")))

    model = BooguImageTransformer2DModel.from_config(cfg)
    sd = read_safetensors_dir(tdir, dtype=mx.float32)
    n, missing, extra = load_named_into_mlx(model, sd)
    print(f"weights: {n} loaded | missing {len(missing)} | extra {len(extra)}")
    if missing:
        print("  MISSING (first 15):", missing[:15])
    # 'extra' is expected to be large (ref_image_* etc. present but unused is fine;
    # but those ARE in the model, so extra should only be non-model tensors)
    if extra:
        print("  EXTRA (first 15):", extra[:15])

    # tiny input
    latent = mx.array(make_seeded_input((1, 16, 16, 16), seed=1))
    timestep = mx.array(np.array([0.5], dtype=np.float32))
    instr = mx.array(make_seeded_input((1, 8, 4096), seed=2))

    out = model(latent, timestep, instr)
    mx.eval(out)
    arr = np.array(out)
    print(f"output shape {arr.shape} | finite {np.isfinite(arr).all()} | "
          f"range ({arr.min():.4f}, {arr.max():.4f}) std {arr.std():.4f}")
    ok = arr.shape == (1, 16, 16, 16) and np.isfinite(arr).all()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
