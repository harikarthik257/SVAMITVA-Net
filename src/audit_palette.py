# MoPR Standards Audit Palette

"""
Provides mapping for the Dual-Roof Classification Logic
to support automated property valuation.
"""

palette_mapping = {
    1: {'label': 'Permanent RCC Roof', 'color': 'White', 'rgb': (255, 255, 255)},
    2: {'label': 'Tiled / Asbestos Roof', 'color': 'Brown', 'rgb': (165, 42, 42)},
    3: {'label': 'Natural Vegetation', 'color': 'Green', 'rgb': (0, 128, 0)},
    0: {'label': 'Open Abadi / Road', 'color': 'Blue', 'rgb': (0, 0, 255)}
}

def get_palette():
    return palette_mapping
