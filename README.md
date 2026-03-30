<div align="center">
  <h1>SVAMITVA-Net</h1>
  <p>An advanced deep learning pipeline designed to automatically segment and extract property boundaries from high-resolution drone imagery, supporting the SVAMITVA scheme's goal of mapping residential land ownership in rural areas.</p>
</div>

---

## Technical Innovations

### Dual-Roof Classification Logic
To aid in the sophisticated automated property valuation defined under MoPR guidelines, SVAMITVA-Net integrates a precise **Dual-Roof Classification Logic**. The system distinctly differentiates:
- **Permanent RCC Roofs** (White)
- **Tiled / Asbestos Roofs** (Brown)

This separation enables accurate economic assessments and standardized categorization of rural property assets directly from aerial features, alongside identifying *Natural Vegetation* and *Open Abadi / Roads*.

## Evidence

<div align="center">
  <h3>Visual Audit & Pipeline Demo</h3>
  <video src="assets/Untitled0.ipynb - Colab - Google Chrome 2026-03-30 19-54-24.mp4" controls="controls" width="800"></video>
  <br><br>
  <h3>Segmentation Results</h3>
  <img src="assets/Screenshot 2026-03-30 194910.png" width="600"/>
  <br><br>
  <img src="assets/Screenshot 2026-03-30 195734.png" width="600"/>
  <br><br>
  <img src="assets/Screenshot 2026-03-30 195800.png" width="600"/>
  <br><br>
  <img src="assets/Screenshot 2026-03-30 195845.png" width="600"/>
  <br><br>
  <img src="assets/Screenshot 2026-03-30 195856.png" width="600"/>
</div>

---

## Metrics
- **Verified IoU Metric**: 95.66%
