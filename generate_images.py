"""
Image Generator: Generate multiple images from a text prompt using Stable Diffusion (diffusers).

Requirements:
  pip install diffusers transformers torch accelerate safetensors
  # For best results, use a machine with a modern NVIDIA GPU and CUDA drivers.

Usage:
  python tools/generate_images.py --prompt "A cat in a space suit" --num 4 --out images/

If you run for the first time, the model will be downloaded automatically (about 4GB).
"""
import argparse
import os
from diffusers import StableDiffusionPipeline
import torch

def main():
    parser = argparse.ArgumentParser(description="Generate images from a text prompt using Stable Diffusion.")
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    parser.add_argument('--num', type=int, default=4, help='Number of images to generate')
    parser.add_argument('--out', type=str, default='images', help='Output folder for images')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading Stable Diffusion pipeline (this may take a while the first time)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to('cuda')
    else:
        print("Warning: No GPU detected. Generation will be very slow on CPU.")

    for i in range(args.num):
        print(f"Generating image {i+1}/{args.num}...")
        image = pipe(args.prompt).images[0]
        out_path = os.path.join(args.out, f"gen_{i+1:02d}.png")
        image.save(out_path)
        print(f"Saved: {out_path}")

    print(f"\nDone! {args.num} images saved to {args.out}/")

if __name__ == "__main__":
    main()
