#!/usr/bin/env python3
"""
A/B test for stable-diffusion-cpp-python GGUF loading.

This script generates two images from the same GGUF file:
  A) using `model_path=...` (often WRONG for GGUF diffusion models)
  B) using `diffusion_model_path=...` (the correct arg for GGUF diffusion models)

It saves both outputs and prints basic pixel statistics so you can quickly tell
if one path produces a flat/grey image.
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


def _stats(img: Image.Image) -> dict[str, float]:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True, help="Path to a GGUF diffusion model file")
    parser.add_argument("--prompt", type=str, default="a lovely cat", help="Prompt text")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument(
        "--keep-vae-on-cpu",
        action="store_true",
        help="Pass keep_vae_on_cpu=True to reduce VRAM usage during decode",
    )
    parser.add_argument(
        "--keep-clip-on-cpu",
        action="store_true",
        help="Pass keep_clip_on_cpu=True to reduce VRAM usage for text encoder",
    )
    parser.add_argument(
        "--also-run-diffusion-model-path",
        action="store_true",
        help="Also run the diffusion_model_path variant (can crash on some builds)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from stable_diffusion_cpp import StableDiffusion
    except Exception as e:
        raise SystemExit(f"Failed to import stable_diffusion_cpp: {e}")

    print(f"GGUF: {args.gguf}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Size: {args.width}x{args.height}, steps={args.steps}, seed={args.seed}")
    if args.keep_vae_on_cpu:
        print("keep_vae_on_cpu=True")
    if args.keep_clip_on_cpu:
        print("keep_clip_on_cpu=True")

    # A) model_path (commonly wrong for GGUF diffusion models)
    sd_a = StableDiffusion(
        model_path=str(args.gguf),
        keep_vae_on_cpu=args.keep_vae_on_cpu,
        keep_clip_on_cpu=args.keep_clip_on_cpu,
    )
    img_a = _as_pil(
        sd_a.generate_image(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            sample_steps=args.steps,
            seed=args.seed,
        )
    )
    path_a = args.out_dir / "out_model_path.png"
    img_a.save(path_a)
    print("A) model_path stats:", _stats(img_a))
    print(f"A) saved: {path_a}")

    if args.also_run_diffusion_model_path:
        # B) diffusion_model_path (can crash on some builds if extra components are required)
        sd_b = StableDiffusion(
            diffusion_model_path=str(args.gguf),
            keep_vae_on_cpu=args.keep_vae_on_cpu,
            keep_clip_on_cpu=args.keep_clip_on_cpu,
        )
        img_b = _as_pil(
            sd_b.generate_image(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                sample_steps=args.steps,
                seed=args.seed,
            )
        )
        path_b = args.out_dir / "out_diffusion_model_path.png"
        img_b.save(path_b)
        print("B) diffusion_model_path stats:", _stats(img_b))
        print(f"B) saved: {path_b}")

    print("\nInterpretation:")
    print("- If A is flat/grey and logs show VAE OOM, enable keep_vae_on_cpu in your app/provider settings.")
    print("- If A is fine with keep_vae_on_cpu, your issue is VRAM pressure during VAE decode, not missing files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
