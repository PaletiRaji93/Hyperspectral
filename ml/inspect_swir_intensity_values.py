
from pathlib import Path

import numpy as np
import spectral


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = ROOT / "dataset" / "raw" / "SWIR" / "13_G"

HDR_PATH = SAMPLE_DIR / "13_G.hdr"

LEAF1_PATH = SAMPLE_DIR / "13_G_leaf1.csv"
LEAF2_PATH = SAMPLE_DIR / "13_G_leaf2.csv"


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("SWIR INTENSITY VALUE INVESTIGATION")
print("=" * 70)

img = spectral.open_image(str(HDR_PATH))

cube = np.asarray(img.load())

leaf1 = np.loadtxt(LEAF1_PATH, delimiter=",", dtype=int)
leaf2 = np.loadtxt(LEAF2_PATH, delimiter=",", dtype=int)

mask1 = np.zeros(cube.shape[:2], dtype=bool)
mask2 = np.zeros(cube.shape[:2], dtype=bool)

mask1[leaf1[:, 0], leaf1[:, 1]] = True
mask2[leaf2[:, 0], leaf2[:, 1]] = True


# ------------------------------------------------------------
# GLOBAL VALUE INVESTIGATION
# ------------------------------------------------------------

print("\nGLOBAL INTENSITY INVESTIGATION")

print("Minimum:", np.min(cube))
print("Maximum:", np.max(cube))

for value in [0, 16383, 65535]:

    count = np.count_nonzero(cube == value)

    percentage = count / cube.size * 100

    print(
        f"Value {value}: "
        f"{count:,} measurements "
        f"({percentage:.6f}%)"
    )


# ------------------------------------------------------------
# LEAF-WISE INVESTIGATION
# ------------------------------------------------------------

print("\nLEAF-WISE INTENSITY INVESTIGATION")

for name, mask in [
    ("Leaf 1", mask1),
    ("Leaf 2", mask2)
]:

    values = cube[mask]

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    print("Measurements:", values.size)

    for value in [0, 16383, 65535]:

        count = np.count_nonzero(values == value)

        print(f"Value {value}: {count:,}")

    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))
    print("Mean:", np.mean(values))

    print(
        "Percentile 99:",
        np.percentile(values, 99)
    )

    print(
        "Percentile 99.9:",
        np.percentile(values, 99.9)
    )


# ------------------------------------------------------------
# FINITE AND HIGH-VALUE INVESTIGATION
# ------------------------------------------------------------

print("\nHIGH INTENSITY INVESTIGATION")

for threshold in [10000, 15000, 16000, 20000, 65535]:

    count = np.count_nonzero(cube >= threshold)

    print(
        f"Measurements >= {threshold}: "
        f"{count:,}"
    )


print("\nINVESTIGATION COMPLETED")
