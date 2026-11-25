
from pathlib import Path

import numpy as np
import pandas as pd
import spectral


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = ROOT / "dataset" / "raw" / "SWIR" / "13_G"

HDR_PATH = SAMPLE_DIR / "13_G.hdr"

LEAF1_PATH = SAMPLE_DIR / "13_G_leaf1.csv"
LEAF2_PATH = SAMPLE_DIR / "13_G_leaf2.csv"

OUTPUT_DIR = ROOT / "artifacts" / "dataset_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("SWIR SATURATION DISTRIBUTION INVESTIGATION")
print("=" * 70)

img = spectral.open_image(str(HDR_PATH))
cube = np.asarray(img.load())

leaf1 = np.loadtxt(LEAF1_PATH, delimiter=",", dtype=int)
leaf2 = np.loadtxt(LEAF2_PATH, delimiter=",", dtype=int)

mask1 = np.zeros(cube.shape[:2], dtype=bool)
mask2 = np.zeros(cube.shape[:2], dtype=bool)

mask1[leaf1[:, 0], leaf1[:, 1]] = True
mask2[leaf2[:, 0], leaf2[:, 1]] = True

saturation = cube == 65535


# ------------------------------------------------------------
# GLOBAL DISTRIBUTION
# ------------------------------------------------------------

print("\nGLOBAL SATURATION DISTRIBUTION")

print("Total saturated measurements:", saturation.sum())

print(
    "Saturated spatial pixels:",
    np.any(saturation, axis=2).sum()
)

print(
    "Spectral bands containing saturation:",
    np.any(saturation, axis=(0, 1)).sum()
)


# ------------------------------------------------------------
# LEAF-WISE DISTRIBUTION
# ------------------------------------------------------------

print("\nLEAF-WISE SATURATION")

for name, mask in [
    ("Leaf 1", mask1),
    ("Leaf 2", mask2)
]:

    leaf_cube = cube[mask]

    count = np.count_nonzero(leaf_cube == 65535)

    print(f"\n{name}")

    print("Annotated pixels:", mask.sum())

    print("Saturated measurements:", count)

    print(
        "Pixels containing saturation:",
        np.any(leaf_cube == 65535, axis=1).sum()
    )

    print(
        "Affected spectral bands:",
        np.any(leaf_cube == 65535, axis=0).sum()
    )


# ------------------------------------------------------------
# BAND-WISE DISTRIBUTION
# ------------------------------------------------------------

band_counts = np.count_nonzero(
    saturation,
    axis=(0, 1)
)

band_df = pd.DataFrame({
    "band_index": np.arange(cube.shape[2]),
    "wavelength_nm": np.asarray(img.bands.centers),
    "saturated_measurements": band_counts
})

band_df.to_csv(
    OUTPUT_DIR / "swir_saturation_by_band.csv",
    index=False
)

print("\nTOP 15 AFFECTED BANDS")

print(
    band_df.sort_values(
        "saturated_measurements",
        ascending=False
    ).head(15).to_string(index=False)
)


# ------------------------------------------------------------
# SPATIAL DISTRIBUTION
# ------------------------------------------------------------

spatial_counts = saturation.sum(axis=2)

rows, cols = np.where(spatial_counts > 0)

print("\nSPATIAL DISTRIBUTION")

if len(rows) > 0:

    print("Affected spatial pixels:", len(rows))

    print("Minimum row:", rows.min())
    print("Maximum row:", rows.max())

    print("Minimum column:", cols.min())
    print("Maximum column:", cols.max())

else:

    print("No spatial pixels contain saturation.")


# ------------------------------------------------------------
# SAVE SPATIAL REPORT
# ------------------------------------------------------------

spatial_df = pd.DataFrame({
    "row": rows,
    "column": cols,
    "saturated_band_count": spatial_counts[rows, cols]
})

spatial_df.to_csv(
    OUTPUT_DIR / "swir_saturation_spatial_report.csv",
    index=False
)


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SATURATION INVESTIGATION COMPLETED")
print("=" * 70)

print("Band report:", OUTPUT_DIR / "swir_saturation_by_band.csv")
print("Spatial report:", OUTPUT_DIR / "swir_saturation_spatial_report.csv")
