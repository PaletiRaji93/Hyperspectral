
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spectral


# ------------------------------------------------------------
# PATHS
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
# LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("SWIR ORIENTATION INVESTIGATION")
print("=" * 70)

img = spectral.open_image(str(HDR_PATH))

cube = np.asarray(img.load())

reference = plt.imread(JPG_PATH)

leaf1 = pd.read_csv(
    LEAF1_PATH,
    header=None
).values.astype(int)

leaf2 = pd.read_csv(
    LEAF2_PATH,
    header=None
).values.astype(int)


# ------------------------------------------------------------
# DIMENSION ANALYSIS
# ------------------------------------------------------------

print("\nDIMENSION ANALYSIS")

print("Cube spatial shape:", cube.shape[:2])
print("Reference image shape:", reference.shape[:2])

print("Reference transpose shape:", reference.T.shape)

print(
    "Reference matches cube directly:",
    reference.shape[:2] == cube.shape[:2]
)

print(
    "Reference transpose matches cube:",
    reference.T.shape == cube.shape[:2]
)


# ------------------------------------------------------------
# BUILD ANNOTATION MASKS
# ------------------------------------------------------------

mask1 = np.zeros(cube.shape[:2], dtype=bool)
mask2 = np.zeros(cube.shape[:2], dtype=bool)

mask1[leaf1[:, 0], leaf1[:, 1]] = True
mask2[leaf2[:, 0], leaf2[:, 1]] = True


# ------------------------------------------------------------
# SELECT SWIR BAND
# ------------------------------------------------------------

band_index = 128

band = cube[:, :, band_index]

print("\nSELECTED BAND:", band_index)

print(
    "Wavelength:",
    float(img.bands.centers[band_index]),
)


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 12)
)

# Original reference image
axes[0, 0].imshow(
    reference,
    cmap="gray"
)

axes[0, 0].set_title(
    "Original SWIR Reference JPG"
)

# Transposed reference image
axes[0, 1].imshow(
    reference.T,
    cmap="gray"
)

axes[0, 1].set_title(
    "Transposed SWIR Reference JPG"
)

# SWIR band with annotations
axes[1, 0].imshow(
    band,
    cmap="gray"
)

axes[1, 0].contour(
    mask1,
    levels=[0.5],
    colors="red",
    linewidths=0.8
)

axes[1, 0].contour(
    mask2,
    levels=[0.5],
    colors="blue",
    linewidths=0.8
)

axes[1, 0].set_title(
    "SWIR Band with Leaf Annotations"
)

# Transposed reference with annotations
axes[1, 1].imshow(
    reference.T,
    cmap="gray"
)

axes[1, 1].contour(
    mask1,
    levels=[0.5],
    colors="red",
    linewidths=0.8
)

axes[1, 1].contour(
    mask2,
    levels=[0.5],
    colors="blue",
    linewidths=0.8
)

axes[1, 1].set_title(
    "Transposed Reference with Leaf Annotations"
)

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

plt.tight_layout()

output_path = (
    OUTPUT_DIR /
    "13_G_swir_orientation_diagnostic.png"
)

plt.savefig(
    output_path,
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("\nVisualization saved:", output_path)

print("\nORIENTATION INVESTIGATION COMPLETED")
