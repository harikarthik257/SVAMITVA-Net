"""
Runs SVAMITVA-Net on a single drone image patch and saves a palette-colored
segmentation mask.

Usage:
    python src/inference.py path/to/image.png -o output_mask.png
    python src/inference.py path/to/patch.npy -o output_mask.png
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from audit_palette import get_palette
from audit import colorize_mask
from model import load_trained_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "weights", "svamitva_refined_v2_e5.pth")


def load_image(path, size=512):
    if path.lower().endswith(".npy"):
        arr = np.load(path)  # expected (3, H, W), uint8
        image = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    else:
        image = Image.open(path).convert("RGB")
    return image.resize((size, size))


def run_inference(model, image):
    array = np.asarray(image).astype(np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).numpy()
    return pred


def main():
    parser = argparse.ArgumentParser(description="SVAMITVA-Net single-image inference")
    parser.add_argument("input", help="Path to an input image (.png/.jpg/.tif) or .npy patch")
    parser.add_argument("-o", "--output", default="prediction.png", help="Path to save the colorized mask")
    parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="Path to model weights")
    args = parser.parse_args()

    model = load_trained_model(args.weights)
    image = load_image(args.input)

    pred_mask = run_inference(model, image)
    colored_mask = colorize_mask(pred_mask, get_palette())

    Image.fromarray(colored_mask).save(args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
