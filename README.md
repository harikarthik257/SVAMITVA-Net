# SVAMITVA-Net

## Project Abstract
SVAMITVA-Net is an advanced deep learning pipeline designed to automatically segment and extract property boundaries from high-resolution drone imagery, supporting the SVAMITVA scheme's goal of mapping residential land ownership in rural areas.

## Metrics
- **Verified IoU Metric**: 95.66%

## Architecture Overview
The core architecture consists of a U-Net model enhanced with a ResNet34 encoder. This robust feature extractor is initialized with ImageNet weights, allowing the network to capture complex spatial dependencies while effectively upsampling to produce precise, high-resolution segmentation masks. It is implemented using `segmentation_models_pytorch`.

## Evidence
Visual demonstrations and outputs of the SVAMITVA-Net pipeline:

### Demo
<video src="assets/Untitled0.ipynb - Colab - Google Chrome 2026-03-30 19-54-24.mp4" controls="controls" style="max-width: 100%;"></video>

### Screenshots
<img src="assets/Screenshot 2026-03-30 194910.png" alt="Evidence 1" width="400"/>
<img src="assets/Screenshot 2026-03-30 195734.png" alt="Evidence 2" width="400"/>
<img src="assets/Screenshot 2026-03-30 195800.png" alt="Evidence 3" width="400"/>
<img src="assets/Screenshot 2026-03-30 195845.png" alt="Evidence 4" width="400"/>
<img src="assets/Screenshot 2026-03-30 195856.png" alt="Evidence 5" width="400"/>
