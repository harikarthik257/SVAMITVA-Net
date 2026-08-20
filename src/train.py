"""
Training script for SVAMITVA-Net (Unet-ResNet34, 5-class semantic segmentation).

Note: this reproduces the architecture and training procedure for the
checkpoint shipped in weights/svamitva_refined_v2_e5.pth, but the original
training run used a larger labeled dataset than the 29-patch demo subset
checked into data/samples/ (which exists for reproducible evaluation, not
full training). Point --data-dir at your full labeled patch set for a real
training run; the default of 5 epochs matches the "_e5" checkpoint naming.

Usage:
    python src/train.py --data-dir data/samples --epochs 5
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import SvamitvaPatchDataset
from model import create_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "samples")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "weights", "svamitva_trained.pth")


def train(data_dir, epochs, batch_size, lr, val_fraction, output_path, device="cpu"):
    dataset = SvamitvaPatchDataset(data_dir)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item() * images.size(0)
        val_loss /= max(len(val_set), 1)

        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"Saved trained weights to {output_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SVAMITVA-Net")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with images/ and masks/ subfolders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.val_fraction, args.output)


if __name__ == "__main__":
    main()
