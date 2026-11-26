
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "outputs" / "spectral_data" / "13_G_spectral_dataset.npz"

OUTPUT_DIR = ROOT / "outputs" / "spectral_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_VALUE = 4095

print("=" * 70)
print("TERRASPECTRA - IGNORE VALUE SPATIAL VISUALIZATION")
print("=" * 70)

# =====================================================
# 2. LOAD EXTRACTED DATA
# =====================================================

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
coords = data["coords"]

print("\nDataset shape:", X.shape)
print("Coordinates:", coords.shape)

# =====================================================
# 3. IDENTIFY AFFECTED PIXELS
# =====================================================

ignore_mask = X == IGNORE_VALUE

ignore_band_count = ignore_mask.sum(axis=1)

affected = ignore_band_count > 0

affected_coords = coords[affected]

affected_labels = y[affected]

print("\nIgnore-value statistics:")
print("Affected pixels:", affected.sum())
print("Affected Leaf 1:", np.sum(affected_labels == 1))
print("Affected Leaf 2:", np.sum(affected_labels == 2))

# =====================================================
# 4. BUILD SPATIAL MAP
# =====================================================

height = int(coords[:, 0].max()) + 1
width = int(coords[:, 1].max()) + 1

spatial_map = np.zeros(
    (height, width),
    dtype=np.uint8
)

for coordinate in affected_coords:
    row, col = coordinate
    spatial_map[row, col] = 1

# =====================================================
# 5. VISUALIZE SPATIAL DISTRIBUTION
# =====================================================

plt.figure(figsize=(14, 7))

plt.imshow(
    spatial_map,
    cmap="hot",
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Ignore-value affected pixel")

plt.xlabel("Column")
plt.ylabel("Row")

plt.title(
    "VNIR Ignore-Value Spatial Distribution - 13_G"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ignore_value_spatial_distribution.png",
    dpi=200
)

plt.close()

# =====================================================
# 6. LEAF-WISE SPATIAL DISTRIBUTION
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, leaf_number in zip(axes, [1, 2]):

    leaf_coords = coords[y == leaf_number]

    leaf_affected_coords = coords[
        (y == leaf_number) & affected
    ]

    ax.scatter(
        leaf_coords[:, 1],
        leaf_coords[:, 0],
        s=1,
        label="Annotated pixels"
    )

    ax.scatter(
        leaf_affected_coords[:, 1],
        leaf_affected_coords[:, 0],
        s=8,
        label="Ignore-value pixels"
    )

    ax.set_title(f"Leaf {leaf_number}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(markerscale=2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "ignore_value_leaf_distribution.png",
    dpi=200
)

plt.close()

# =====================================================
# 7. IGNORE VALUE COUNT PER PIXEL
# =====================================================

count_table = pd.DataFrame({
    "Row": coords[:, 0],
    "Column": coords[:, 1],
    "Leaf_Label": y,
    "Ignore_Band_Count": ignore_band_count
})

count_table.to_csv(
    OUTPUT_DIR / "ignore_pixel_spatial_report.csv",
    index=False
)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\n" + "=" * 70)
print("SPATIAL VISUALIZATION COMPLETED")
print("=" * 70)
