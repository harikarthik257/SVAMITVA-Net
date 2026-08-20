"""
Runs SVAMITVA-Net on a georeferenced drone GeoTIFF and vectorizes the
per-class prediction into a GeoPackage of polygons -- the output format
(GPKG) called for by the MoPR SVAMITVA hackathon's OGC feature-extraction
problem statement, as opposed to a flat image mask.

Usage:
    python src/vectorize.py path/to/input.tif -o output.gpkg

NOTE: unlike the other scripts in this repo, this one has not been run
end-to-end in development -- rasterio/geopandas were unavailable in the
environment used to build this repo (compiled extensions incompatible with
the Python version installed there). Install rasterio, geopandas and
shapely (see requirements.txt) and smoke-test this against a real GeoTIFF
before relying on it.
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from audit_palette import get_palette
from model import load_trained_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "weights", "svamitva_refined_v2_e5.pth")


def read_raster_as_model_input(path, size=512):
    """
    Reads a GeoTIFF and returns (image_array, transform, crs). If the raster
    is single-band (as the hackathon-provided sample raster is), the band is
    replicated across 3 channels since the model expects RGB input.
    """
    with rasterio.open(path) as src:
        bands = src.read()  # (bands, H, W)
        transform = src.transform
        crs = src.crs

    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    elif bands.shape[0] > 3:
        bands = bands[:3]

    return bands, transform, crs


def predict_mask(model, bands):
    import torch

    image = bands.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).numpy()
    return pred.astype(np.int32)


def mask_to_geodataframe(mask, transform, crs, palette, skip_background=True):
    records = []
    for geom, class_id in shapes(mask, transform=transform):
        class_id = int(class_id)
        if skip_background and class_id == 0:
            continue
        label = palette.get(class_id, {}).get("label", f"class_{class_id}")
        records.append({"class_id": class_id, "label": label, "geometry": shape(geom)})

    if not records:
        return gpd.GeoDataFrame(columns=["class_id", "label", "geometry"], geometry="geometry", crs=crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def main():
    parser = argparse.ArgumentParser(description="Vectorize a SVAMITVA-Net prediction into a GeoPackage")
    parser.add_argument("input", help="Path to an input GeoTIFF")
    parser.add_argument("-o", "--output", default="prediction.gpkg", help="Path to save the output GeoPackage")
    parser.add_argument("-w", "--weights", default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--keep-background", action="store_true", help="Include background polygons in the output")
    args = parser.parse_args()

    model = load_trained_model(args.weights)
    bands, transform, crs = read_raster_as_model_input(args.input)
    mask = predict_mask(model, bands)

    gdf = mask_to_geodataframe(mask, transform, crs, get_palette(), skip_background=not args.keep_background)
    gdf.to_file(args.output, driver="GPKG")

    print(f"Wrote {len(gdf)} polygons to {args.output}")


if __name__ == "__main__":
    main()
