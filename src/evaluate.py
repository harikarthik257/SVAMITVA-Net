"""
Computes per-class and mean IoU for SVAMITVA-Net against the labeled sample
patches in data/samples/. This is the script used to produce the IoU figures
quoted in the README -- run it yourself to reproduce them.

Usage:
    python src/evaluate.py
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from audit_palette import get_palette
from dataset import SvamitvaPatchDataset
from model import load_trained_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(REPO_ROOT, "weights", "svamitva_refined_v2_e5.pth")
SAMPLES_DIR = os.path.join(REPO_ROOT, "data", "samples")


def compute_iou(model, dataset, num_classes=5):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for image, mask in loader:
            logits = model(image)
            pred = torch.argmax(logits, dim=1).squeeze(0).numpy()
            mask = mask.squeeze(0).numpy()

            for c in range(num_classes):
                p, g = (pred == c), (mask == c)
                intersection[c] += np.logical_and(p, g).sum()
                union[c] += np.logical_or(p, g).sum()

    return intersection / np.maximum(union, 1)


def main():
    model = load_trained_model(WEIGHTS_PATH)
    dataset = SvamitvaPatchDataset(SAMPLES_DIR)
    palette = get_palette()

    iou_per_class = compute_iou(model, dataset)

    print(f"Evaluated on {len(dataset)} labeled patches from data/samples/\n")
    for class_id, iou in enumerate(iou_per_class):
        label = palette[class_id]["label"]
        print(f"  class {class_id} ({label:>10}): IoU = {iou * 100:.2f}%")

    print(f"\nMean IoU (all {len(iou_per_class)} classes):        {iou_per_class.mean() * 100:.2f}%")
    print(f"Mean IoU (excluding background):    {iou_per_class[1:].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
