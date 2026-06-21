"""TI2I conditioning golden (torch): Qwen3-VL last_hidden_state with input image.

Saves processor outputs (input_ids, pixel_values, image_grid_thw) + feats so the
MLX side can run the identical multimodal inputs and parity-check the forward.
"""
import os
import numpy as np

BASE = os.path.expanduser(os.environ.get("BOOGU_BASE",
        "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
IMG = os.path.expanduser(os.environ.get("BOOGU_EDIT_IMG",
        "~/Development/mlxengine-image/boogu-image-mlx/samples/pure_mlx_cabin.png"))
INSTR = os.environ.get("BOOGU_EDIT_INSTR", "change the scene to a bright sunny day")
SYSTEM_TI2I = ("Describe the key features of the input image (color, shape, size, texture, objects, "
               "background), then explain how the user's text instruction should alter or modify the "
               "image. Generate a new image that meets the user's requirements while maintaining "
               "consistency with the original input where appropriate.")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edit_conditioning_golden.npz")


def main() -> int:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(os.path.join(BASE, "processor"))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(BASE, "mllm"), torch_dtype=torch.float32).eval()
    base = model.model

    img = Image.open(IMG).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TI2I}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": INSTR}]},
    ]
    inputs = processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True, add_generation_prompt=False,
        padding=True, return_tensors="pt")
    with torch.no_grad():
        try:
            feats = base(**inputs, output_hidden_states=False).last_hidden_state
        except Exception:
            feats = base(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]

    save = {"feats": feats.detach().cpu().float().numpy(),
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "pixel_values": inputs["pixel_values"].cpu().float().numpy(),
            "image_grid_thw": inputs["image_grid_thw"].cpu().numpy(),
            "instr": np.array(INSTR)}
    print("input_ids", save["input_ids"].shape, "pixel_values", save["pixel_values"].shape,
          "grid", save["image_grid_thw"].tolist(), "feats", save["feats"].shape)
    np.savez(OUT, **save)
    print("saved ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
