#!/usr/bin/env python

"""
Extract the VLM (SmolVLM) backbone from a SmolVLA checkpoint and save it as a
standalone Hugging Face model for easy comparison.

Usage:
  python examples/Bluekiwi/extract_vlm.py \
    --smolvla models/smolvla_fixed \
    --out outputs/extracted_vlm \
    --base_vlm HuggingFaceTB/SmolVLM2-500M-Video-Instruct

Offline:
  export HF_HUB_OFFLINE=1
  # Ensure base_vlm is cached locally or provide a local path to it.
"""

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
)


def resolve_base_vlm_path(arg_val: str) -> str:
    # Allow overriding via environment variable if needed
    env_path = os.environ.get("SMOLVLM2_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    return arg_val


def load_smolvla_state_dict(smolvla_dir: Path) -> dict[str, torch.Tensor]:
    safetensor = smolvla_dir / "model.safetensors"
    if not safetensor.exists():
        raise FileNotFoundError(f"model.safetensors not found in {smolvla_dir}")
    return load_file(str(safetensor))


def extract_vlm_subdict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Extract only the VLM submodule weights from SmolVLA policy state_dict.

    Looks for keys containing 'vlm_with_expert.vlm.' and strips that prefix.
    """
    prefix = "vlm_with_expert.vlm."
    alt_prefix = ".vlm_with_expert.vlm."
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if prefix in k:
            new_k = k.split(prefix, 1)[1]
            out[new_k] = v
        elif alt_prefix in k:
            new_k = k.split(alt_prefix, 1)[1]
            out[new_k] = v
    if not out:
        raise RuntimeError(
            "Could not find any VLM keys in SmolVLA state_dict."
        )
    return out


def main():
    parser = argparse.ArgumentParser("Extract VLM from SmolVLA")
    parser.add_argument("--smolvla", default="models/smolvla_fixed", help="Path to SmolVLA checkpoint dir")
    parser.add_argument(
        "--base_vlm",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Base SmolVLM model id or local path (for architecture + processor)",
    )
    parser.add_argument("--out", default="outputs/extracted_vlm", help="Output directory for extracted VLM")
    args = parser.parse_args()

    smolvla_dir = Path(args.smolvla)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] SmolVLA dir: {smolvla_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    # 1) Load SmolVLA state dict
    sd_smolvla = load_smolvla_state_dict(smolvla_dir)
    sd_vlm = extract_vlm_subdict(sd_smolvla)
    print(f"[INFO] Extracted VLM tensors: {len(sd_vlm)}")

    # 2) Load base VLM model (architecture)
    base_vlm_path = resolve_base_vlm_path(args.base_vlm)
    print(f"[INFO] Base VLM source: {base_vlm_path}")
    try:
        processor = AutoProcessor.from_pretrained(base_vlm_path, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load processor from {base_vlm_path}. If offline, set SMOLVLM2_PATH to a local directory.\n{e}"
        )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_vlm_path,
            local_files_only=True,
            device_map="cpu",
        )
    except Exception:
        # Fallback: build from config
        cfg = AutoConfig.from_pretrained(base_vlm_path, local_files_only=True)
        model = AutoModelForImageTextToText.from_config(cfg)

    # 3) Load extracted weights (strip prefix already)
    missing, unexpected = model.load_state_dict(sd_vlm, strict=False)
    print(f"[INFO] Loaded VLM weights into base model")
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}), showing first 10:")
        for k in missing[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}), showing first 10:")
        for k in unexpected[:10]:
            print(f"  - {k}")

    # 4) Save as standalone HF model (safetensors) + processor
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    print(f"[OK] Saved extracted VLM to {out_dir}")


if __name__ == "__main__":
    main()
