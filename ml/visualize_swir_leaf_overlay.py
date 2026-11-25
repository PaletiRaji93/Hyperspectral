
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral


# ============================================================
# TERRASPECTRA - SWIR VISUALIZATION AND QUALITY VALIDATION
# ============================================================

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = ROOT / "dataset" / "raw" / "SWIR" / "13_G"

HDR_PATH = SAMPLE_DIR / "13_G.hdr"
JPG_PATH = SAMPLE_DIR / "13_G.jpg"

LEAF1_PATH = SAMPLE_DIR / "13_G_leaf1.csv"
LEAF2_PATH = SAMPLE_DIR / "13_G_leaf2.csv"

OUTPUT_DIR = ROOT / "outputs" / "swir_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. VERIFY INPUT FILES
# ------------------------------------------------------------

required_files = [
    HDR_PATH,
    JPG_PATH,
    LEAF1_PATH,
    LEAF2_PATH
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


# ------------------------------------------------------------
# 3. LOAD SWIR HYPERSPECTRAL IMAGE
# ------------------------------------------------------------

print("=" * 70)
print("TERRASPECTRA - SWIR VISUALIZATION")
print("=" * 70)

img = spectral.open_image(str(HDR_PATH))

cube = np.asarray(img.load())

height, width, bands = cube.shape

print("\nSWIR CUBE INFORMATION")
print("-" * 50)

print("Shape:", cube.shape)
print("Height:", height)
print("Width:", width)
print("Spectral bands:", bands)
print("Data type:", cube.dtype)


# ------------------------------------------------------------
# 4. LOAD LEAF ANNOTATIONS
# ------------------------------------------------------------

leaf1 = pd.read_csv(
    LEAF1_PATH,
    header=None
).values.astype(int)

leaf2 = pd.read_csv(
    LEAF2_PATH,
    header=None
).values.astype(int)

print("\nLEAF ANNOTATION INFORMATION")
print("-" * 50)

print("Leaf 1 coordinates:", leaf1.shape)
print("Leaf 2 coordinates:", leaf2.shape)


# ------------------------------------------------------------
# 5. VALIDATE COORDINATES
# ------------------------------------------------------------

def validate_coordinates(coordinates, name):

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"{name} must contain exactly two columns: row and column."
        )

    if len(coordinates) == 0:
        raise ValueError(f"{name} contains no coordinates.")

    rows = coordinates[:, 0]
    cols = coordinates[:, 1]

    valid = (
        (rows >= 0) &
        (rows < height) &
        (cols >= 0) &
        (cols < width)
    )

    print(f"\n{name} coordinate validation:")
    print("Total coordinates:", len(coordinates))
    print("Valid coordinates:", np.sum(valid))
    print("Invalid coordinates:", np.sum(~valid))

    if not np.all(valid):
        raise ValueError(
            f"{name} contains coordinates outside the SWIR image."
        )


validate_coordinates(leaf1, "Leaf 1")
validate_coordinates(leaf2, "Leaf 2")


# ------------------------------------------------------------
# 6. CREATE LEAF MASKS
# ------------------------------------------------------------

mask1 = np.zeros((height, width), dtype=bool)
mask2 = np.zeros((height, width), dtype=bool)

mask1[leaf1[:, 0], leaf1[:, 1]] = True
mask2[leaf2[:, 0], leaf2[:, 1]] = True

print("\nMASK INFORMATION")
print("-" * 50)

print("Leaf 1 mask pixels:", np.sum(mask1))
print("Leaf 2 mask pixels:", np.sum(mask2))

overlap = np.sum(mask1 & mask2)

print("Overlapping pixels:", overlap)

if overlap > 0:
    print("WARNING: Leaf masks contain overlapping pixels.")
else:
    print("Leaf masks do not overlap.")


# ------------------------------------------------------------
# 7. LOAD REFERENCE IMAGE
# ------------------------------------------------------------

reference = plt.imread(JPG_PATH)

print("\nREFERENCE IMAGE")
print("-" * 50)

print("Reference image shape:", reference.shape)


# ------------------------------------------------------------
# 8. SELECT REPRESENTATIVE SWIR BAND
# ------------------------------------------------------------

# Band 128 (zero-based index) is used for visualization.
band_index = 128

band_image = cube[:, :, band_index].astype(np.float32)

print("\nSELECTED SWIR BAND")
print("-" * 50)

print("Band index:", band_index)

if len(img.bands.centers) > band_index:
    print(
        "Wavelength (nm):",
        img.bands.centers[band_index]
    )


# ------------------------------------------------------------
# 9. GENERATE VISUALIZATIONS
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 11)
)

axes[0, 0].imshow(reference)
axes[0, 0].set_title("SWIR Reference JPG")

axes[0, 1].imshow(mask1, cmap="gray")
axes[0, 1].set_title("SWIR Leaf 1 Mask")

axes[1, 0].imshow(mask2, cmap="gray")
axes[1, 0].set_title("SWIR Leaf 2 Mask")

axes[1, 1].imshow(
    band_image,
    cmap="gray"
)

axes[1, 1].imshow(
    mask1,
    cmap="Reds",
    alpha=0.35
)

axes[1, 1].imshow(
    mask2,
    cmap="Blues",
    alpha=0.35
)

axes[1, 1].set_title(
    f"SWIR Band {band_index} with Leaf Masks"
)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

output_path = (
    OUTPUT_DIR / "13_G_swir_leaf_visualization.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nVisualization saved:", output_path)

plt.close()


# ------------------------------------------------------------
# 10. LEAF MASK BOUNDING BOX VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SWIR LEAF MASK VALIDATION")
print("=" * 70)

for name, mask in [
    ("Leaf 1", mask1),
    ("Leaf 2", mask2)
]:

    rows, cols = np.where(mask)

    print(f"\n{name}")

    print("Pixel count:", len(rows))

    print(
        "Row range:",
        rows.min(),
        "-",
        rows.max()
    )

    print(
        "Column range:",
        cols.min(),
        "-",
        cols.max()
    )


# ------------------------------------------------------------
# 11. BAND 128 STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SWIR BAND STATISTICS")
print("=" * 70)

finite_band = band_image[np.isfinite(band_image)]

print("\nBand index:", band_index)

print("Minimum:", np.min(finite_band))
print("Maximum:", np.max(finite_band))
print("Mean:", np.mean(finite_band))
print("Standard deviation:", np.std(finite_band))


# ------------------------------------------------------------
# 12. SPECTRAL DATA QUALITY VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SWIR DATA QUALITY VALIDATION")
print("=" * 70)

nan_count = np.isnan(cube).sum()
infinite_count = np.isinf(cube).sum()

# SWIR ignore value from the HDR metadata
ignore_value = 16383

ignore_count = np.sum(cube == ignore_value)

zero_count = np.sum(cube == 0)

print("\nNaN values:", nan_count)
print("Infinite values:", infinite_count)

print("Ignore value:", ignore_value)
print("Ignore-value measurements:", ignore_count)

print("Zero measurements:", zero_count)

print(
    "Total measurements:",
    cube.size
)

print(
    "Finite measurements:",
    np.isfinite(cube).sum()
)


# ------------------------------------------------------------
# 13. LEAF SPECTRAL MEASUREMENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SWIR LEAF SPECTRAL SUMMARY")
print("=" * 70)

for name, mask in [
    ("Leaf 1", mask1),
    ("Leaf 2", mask2)
]:

    leaf_pixels = cube[mask]

    finite_values = leaf_pixels[
        np.isfinite(leaf_pixels)
    ]

    print(f"\n{name}")

    print("Pixel count:", leaf_pixels.shape[0])
    print("Spectral bands:", leaf_pixels.shape[1])

    print("Mean intensity:", np.mean(finite_values))
    print("Minimum intensity:", np.min(finite_values))
    print("Maximum intensity:", np.max(finite_values))


# ------------------------------------------------------------
# 14. FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SWIR INSPECTION COMPLETED")
print("=" * 70)

print("Sample: 13_G")
print("Sensor: SWIR")
print("Output:", output_path)
print("Leaf masks: Generated")
print("Quality validation: Completed")
