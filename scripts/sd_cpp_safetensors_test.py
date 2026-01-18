#!/usr/bin/env python3
"""
Sanity test for stable-diffusion-cpp-python with a single-file safetensors checkpoint.

This mirrors the stable-diffusion.cpp README usage: pass the safetensors file
as `model_path=...`, generate, and save an image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _as_pil(result) -> Image.Image:
    if isinstance(result, list):
        result = result[0]
    if not isinstance(result, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to a .safetensors SD checkpoint")
    parser.add_argument("--prompt", type=str, default="a lovely cat", help="Prompt text")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("out_safetensors.png"))
    args = parser.parse_args()

    try:
        from stable_diffusion_cpp import StableDiffusion
    except Exception as e:
        raise SystemExit(f"Failed to import stable_diffusion_cpp: {e}")

    sd = StableDiffusion(model_path=str(args.model))
    img = _as_pil(
        sd.generate_image(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            sample_steps=args.steps,
            seed=args.seed,
        )
    )
    img.save(args.out)

    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    print(f"Saved: {args.out}")
    print(
        {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

