import torch
import matplotlib.pyplot as plt
import numpy as np

from audit_palette import get_palette

def colorize_mask(mask, palette=None):
    """
    Maps a (H, W) integer class mask to an (H, W, 3) RGB image using the
    SVAMITVA-Net class palette.
    """
    if palette is None:
        palette = get_palette()
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, info in palette.items():
        rgb[mask == class_id] = info['rgb']
    return rgb

def perform_visual_audit(input_image, ground_truth, prediction, save_path=None):
    """
    Performs a 3-panel visual audit comparing Input, Ground Truth, and Prediction.
    Ground truth and prediction masks are rendered with the SVAMITVA-Net class
    palette (audit_palette.py) rather than raw grayscale, so the panels are
    directly comparable to the class legend.

    Args:
        input_image (torch.Tensor or np.ndarray): The input image (C, H, W) or (H, W, C).
        ground_truth (torch.Tensor or np.ndarray): The ground truth mask (H, W).
        prediction (torch.Tensor or np.ndarray): The model's prediction mask (H, W).
        save_path (str, optional): Path to save the audit image. If None, it will be displayed.
    """
    if isinstance(input_image, torch.Tensor):
        input_image = input_image.cpu().detach().numpy()
    if input_image.ndim == 3 and input_image.shape[0] in [1, 3]: # C, H, W to H, W, C
        input_image = np.transpose(input_image, (1, 2, 0))
        if input_image.shape[2] == 1:
            input_image = input_image.squeeze(2)

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().detach().numpy()
        if ground_truth.ndim == 3 and ground_truth.shape[0] == 1:
            ground_truth = ground_truth.squeeze(0)

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)

    # Normalize input image for display
    if input_image.max() > 1.0 and input_image.dtype != np.uint8:
        input_image = input_image / input_image.max()

    palette = get_palette()
    ground_truth_rgb = colorize_mask(ground_truth.astype(np.int64), palette)
    prediction_rgb = colorize_mask(prediction.astype(np.int64), palette)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cmap_img = 'gray' if input_image.ndim == 2 else None
    axes[0].imshow(input_image, cmap=cmap_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(ground_truth_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(prediction_rgb)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=np.array(info['rgb']) / 255.0, label=info['label'])
        for info in palette.values()
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(palette), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visual audit saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":
    pass
