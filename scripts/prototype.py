import os
import yaml
import subprocess
import argparse
from PIL import Image
from pathlib import Path

CHARACTER_DIR = Path("../character_yamls")
OUTPUT_DIR = Path("../sketches")

def load_character(character_name):
    file_path = CHARACTER_DIR / f"{character_name}.yaml"
    if not file_path.exists():
        raise FileNotFoundError(f"Character YAML not found: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def combine_prompts(character_data, mode):
    prompt_parts = character_data["prompts"].get(mode)
    if not prompt_parts:
        raise ValueError(f"No prompts found for mode: {mode}")
    return ", ".join(prompt_parts)

def generate_images(prompt, outdir, resolution, count, mode):
    os.makedirs(outdir, exist_ok=True)
    for i in range(count):
        print(f"→ Generating image {i+1}/{count}")
        subprocess.run([
            "invokeai",
            "--prompt", prompt,
            "--model", "dreamshaper_8",
            "--width", str(resolution),
            "--height", str(resolution),
            "--num_images", "1",
            "--outdir", str(outdir)
        ])
        if mode == "grayscale":
            grayscale_latest(outdir)

def grayscale_latest(directory):
    files = sorted(Path(directory).glob("*.png"), key=os.path.getmtime)
    if files:
        latest = files[-1]
        img = Image.open(latest).convert("L")
        img.save(latest)
        print(f"✓ Grayscaled {latest.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", required=True)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--mode", choices=["grayscale", "color"], default="grayscale")
    args = parser.parse_args()

    char = load_character(args.character)
    prompt = combine_prompts(char, args.mode)
    outdir = OUTPUT_DIR / args.character
    generate_images(prompt, outdir, args.resolution, args.count, args.mode)

if __name__ == "__main__":
    main()
