"""Produce a golden instruction-feature tensor from Boogu's mllm (torch oracle).

Replicates pipeline_boogu._get_instruction_feature_embeds for the Base T2I path
(no prompt tuning, num_instruction_feature_layers=1, no images):

  messages = [system(T2I), user(instruction)]
  inputs   = processor.apply_chat_template(..., tokenize=True, return_dict=True)
  feats    = mllm.model(**inputs).last_hidden_state    # [1, L, 4096]

Saves feats + input_ids to conditioning_golden.npz for e2e + MLX-parity.
"""

import os
import sys

import numpy as np

BASE = os.path.expanduser(os.environ.get("BOOGU_BASE",
        "~/Development/mlxengine-image/weights/Boogu-Image-0.1-Base"))
PROMPT = os.environ.get("BOOGU_PROMPT", "a red panda surfing on a wave, photorealistic")
SYSTEM_T2I = ("You are a helpful assistant that generates high-quality images based "
              "on user instructions. The instructions are as follows.")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conditioning_golden.npz")


def main() -> int:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    mllm_dir = os.path.join(BASE, "mllm")
    proc_dir = os.path.join(BASE, "processor")

    processor = AutoProcessor.from_pretrained(proc_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        mllm_dir, torch_dtype=torch.float32).eval()
    base = model.model  # strip lm_head -> base Qwen3VLModel (pipeline does mllm.model)

    def extract(text: str) -> np.ndarray:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_T2I}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        inputs = processor.apply_chat_template(
            [messages], padding="longest", padding_side="right",
            return_tensors="pt", tokenize=True, return_dict=True,
            add_generation_prompt=False,
        )
        with torch.no_grad():
            try:
                feats = base(**inputs, output_hidden_states=False).last_hidden_state
            except Exception:
                feats = base(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1]
        return feats.detach().cpu().float().numpy(), inputs["input_ids"].cpu().numpy()

    feats_pos, ids_pos = extract(PROMPT)               # positive
    feats_neg, ids_neg = extract("")                   # negative (empty -> CFG uncond)

    print(f"prompt: {PROMPT!r}")
    print(f"pos feats {feats_pos.shape} range ({feats_pos.min():.2f},{feats_pos.max():.2f}) | "
          f"neg feats {feats_neg.shape}")
    np.savez(OUT, feats=feats_pos, feats_neg=feats_neg,
             input_ids=ids_pos, input_ids_neg=ids_neg, prompt=np.array(PROMPT))
    print("saved ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
