"""
Interactive Gradio demo for SVAMITVA-Net: pick a labeled sample patch, run
the model, and see Input / Ground Truth / Prediction side by side with a
per-class IoU breakdown.

Usage:
    python src/app.py

NOTE: gradio was unavailable in the environment used to build this repo (no
network access to install it), so the UI wiring below (gr.Blocks layout,
gallery selection callback) has not been execution-tested, unlike the
audit_result() logic it calls, which reuses load_trained_model/run_inference/
colorize_mask exactly as evaluate.py and inference.py do (both verified).
Install gradio (see requirements.txt) and smoke-test this before demoing it.
"""
import os

import gradio as gr
import numpy as np
from PIL import Image

from audit import colorize_mask
from audit_palette import get_palette
from inference import run_inference
from model import load_trained_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(REPO_ROOT, "weights", "svamitva_refined_v2_e5.pth")
IMAGES_DIR = os.path.join(REPO_ROOT, "data", "samples", "images")
MASKS_DIR = os.path.join(REPO_ROOT, "data", "samples", "masks")

PALETTE = get_palette()
MODEL = load_trained_model(WEIGHTS_PATH)
SAMPLE_FILES = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".npy"))


def load_rgb_array(patch_filename):
    array = np.load(os.path.join(IMAGES_DIR, patch_filename))  # (3, H, W) uint8
    return np.transpose(array, (1, 2, 0))  # (H, W, 3) uint8


def per_class_iou_report(pred_mask, gt_mask):
    lines, ious = [], []
    for class_id, info in PALETTE.items():
        pred_c, gt_c = (pred_mask == class_id), (gt_mask == class_id)
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        iou = np.logical_and(pred_c, gt_c).sum() / union
        ious.append(iou)
        lines.append(f"{info['label']}: {iou * 100:.1f}%")
    mean_iou = float(np.mean(ious)) * 100 if ious else 0.0
    return f"Mean IoU: {mean_iou:.2f}%  |  " + "  ".join(lines)


def run_audit(evt: gr.SelectData):
    filename = SAMPLE_FILES[evt.index]
    image_array = load_rgb_array(filename)
    gt_mask = np.load(os.path.join(MASKS_DIR, filename))

    pred_mask = run_inference(MODEL, Image.fromarray(image_array))

    gt_rgb = colorize_mask(gt_mask, PALETTE)
    pred_rgb = colorize_mask(pred_mask, PALETTE)
    status = per_class_iou_report(pred_mask, gt_mask)

    return image_array, gt_rgb, pred_rgb, status


def build_demo():
    gallery_images = [load_rgb_array(f) for f in SAMPLE_FILES]

    with gr.Blocks() as demo:
        gr.HTML(
            "<div style='text-align:center; background:#1e3c72; color:white; "
            "padding:10px; border-radius:10px;'>"
            "<h1>SVAMITVA-Net Audit Demo</h1></div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    value=list(zip(gallery_images, SAMPLE_FILES)),
                    label="Select a labeled patch",
                    columns=4,
                    height=400,
                )
                status_box = gr.Textbox(label="Per-class IoU")

            with gr.Column(scale=2):
                with gr.Row():
                    out_input = gr.Image(label="Input")
                    out_gt = gr.Image(label="Ground Truth")
                    out_pred = gr.Image(label="Prediction")

        gallery.select(run_audit, None, [out_input, out_gt, out_pred, status_box])

    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft())
