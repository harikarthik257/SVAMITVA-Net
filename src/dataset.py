"""
PyTorch Dataset for SVAMITVA-Net's labeled patch data (matched .npy image/mask
pairs, as produced by the drone patch-extraction pipeline).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SvamitvaPatchDataset(Dataset):
    """
    Loads (image, mask) pairs from a directory of the form:
        root/images/<name>.npy   -- (3, H, W) uint8, RGB
        root/masks/<name>.npy    -- (H, W) uint8, class ids 0-4

    Images are normalized to [0, 1] by dividing by 255; masks are returned
    as int64 class-id maps, ready for nn.CrossEntropyLoss.
    """

    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.filenames = sorted(
            f for f in os.listdir(self.images_dir) if f.endswith(".npy")
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        image = np.load(os.path.join(self.images_dir, name)).astype(np.float32) / 255.0
        mask = np.load(os.path.join(self.masks_dir, name)).astype(np.int64)

        return torch.from_numpy(image), torch.from_numpy(mask)
