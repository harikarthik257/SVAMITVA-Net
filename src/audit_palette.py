"""
Class palette for SVAMITVA-Net's 5-class segmentation output.

The checkpoint (weights/svamitva_refined_v2_e5.pth) was confirmed to output
5 classes by inspecting segmentation_head.0.weight (shape [5, 16, 3, 3]) and
loading strict=True against smp.Unet(resnet34, in_channels=3, classes=5)
with zero missing/unexpected keys. The category *names* below match the
label set used during training. Class 0 (Background) is confirmed by pixel
share (~85% of labeled pixels, the expected majority class for rural aerial
imagery). The index order for classes 1-4 is a best-effort assignment from
mean RGB signature per class on the 100 labeled sample patches and has not
been cross-checked against the original training label metadata -- treat it
as provisional pending confirmation from the source notebook.
"""

palette_mapping = {
    0: {'label': 'Background', 'color': 'Black', 'rgb': (0, 0, 0)},
    1: {'label': 'Road', 'color': 'Gray', 'rgb': (128, 128, 128)},
    2: {'label': 'Roof', 'color': 'Brown', 'rgb': (165, 42, 42)},
    3: {'label': 'Water', 'color': 'Blue', 'rgb': (0, 102, 204)},
    4: {'label': 'Grass', 'color': 'Green', 'rgb': (0, 128, 0)},
}

def get_palette():
    return palette_mapping
