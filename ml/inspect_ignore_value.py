
from pathlib import Path
import numpy as np
import pandas as pd
from spectral import open_image

# =====================================================
# 1. PATHS
# =====================================================

DATA_PATH = Path("outputs/spectral_data/13_G_spectral_dataset.npz")
HDR_PATH = Path("dataset/raw/VNIR/13_G/13_G.hdr")

OUTPUT = Path("artifacts/spectral_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

IGNORE_VALUE = 4095

print("=" * 70)
print("TERRASPECTRA - ENVI IGNORE VALUE INVESTIGATION")
print("=" * 70)

# =====================================================
# 2. LOAD DATA
# =====================================================

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
wavelengths = data["wavelengths"]
coordinates = data["coords"]

image = open_image(str(HDR_PATH))
metadata = image.metadata

print("\n1. DATA INFORMATION")
print("Shape:", X.shape)
print("Data type:", X.dtype)
print("ENVI ignore value:", metadata.get("data ignore value"))

# =====================================================
# 3. CHECK IGNORE VALUE
# =====================================================

ignore_mask = X == IGNORE_VALUE

total_ignore = np.count_nonzero(ignore_mask)

print("\n2. IGNORE VALUE STATISTICS")

print("Ignore value:", IGNORE_VALUE)
print("Total occurrences:", total_ignore)
print("Percentage:", total_ignore / X.size * 100)

# =====================================================
# 4. LEAF-WISE INVESTIGATION
# =====================================================

leaf_results = []

for label in [1, 2]:

    leaf = X[y == label]

    mask = leaf == IGNORE_VALUE

    count = np.count_nonzero(mask)

    affected_pixels = np.any(mask, axis=1).sum()

    print(f"\nLeaf {label}")
    print("Total pixels:", len(leaf))
    print("Ignore occurrences:", count)
    print("Affected pixels:", affected_pixels)

    print(
        "Affected pixel percentage:",
        affected_pixels / len(leaf) * 100
    )

    leaf_results.append({
        "Leaf": label,
        "Ignore_Count": count,
        "Affected_Pixels": affected_pixels,
        "Affected_Percentage":
            affected_pixels / len(leaf) * 100
    })

pd.DataFrame(leaf_results).to_csv(
    OUTPUT / "ignore_value_leaf_report.csv",
    index=False
)

# =====================================================
# 5. IGNORE VALUE DISTRIBUTION PER BAND
# =====================================================

band_counts = np.count_nonzero(
    X == IGNORE_VALUE,
    axis=0
)

leaf1 = X[y == 1]
leaf2 = X[y == 2]

band_report = pd.DataFrame({
    "Band": np.arange(X.shape[1]),
    "Wavelength_nm": wavelengths,
    "Leaf1_Ignore_Count":
        np.count_nonzero(leaf1 == IGNORE_VALUE, axis=0),
    "Leaf2_Ignore_Count":
        np.count_nonzero(leaf2 == IGNORE_VALUE, axis=0),
    "Combined_Ignore_Count": band_counts,
    "Ignore_Percentage":
        band_counts / len(X) * 100
})

print("\n3. TOP 15 BANDS CONTAINING IGNORE VALUE")

print(
    band_report.nlargest(
        15, "Ignore_Percentage"
    ).to_string(index=False)
)

band_report.to_csv(
    OUTPUT / "ignore_value_band_report.csv",
    index=False
)

# =====================================================
# 6. PIXEL-LEVEL INVESTIGATION
# =====================================================

ignore_per_pixel = np.count_nonzero(
    X == IGNORE_VALUE,
    axis=1
)

print("\n4. PIXEL-LEVEL STATISTICS")

print("Maximum ignore values in one pixel:",
      ignore_per_pixel.max())

print("Pixels containing at least one ignore value:",
      np.count_nonzero(ignore_per_pixel > 0))

for threshold in [1, 2, 5, 10, 20]:

    print(
        f"Pixels with at least {threshold} ignore bands:",
        np.count_nonzero(ignore_per_pixel >= threshold)
    )

# =====================================================
# 7. SAVE AFFECTED PIXEL COORDINATES
# =====================================================

affected_indices = np.where(ignore_per_pixel > 0)[0]

affected_coordinates = coordinates[affected_indices]

affected_report = pd.DataFrame({
    "Row": affected_coordinates[:, 0],
    "Column": affected_coordinates[:, 1],
    "Leaf_Label": y[affected_indices],
    "Ignore_Band_Count": ignore_per_pixel[affected_indices]
})

affected_report.to_csv(
    OUTPUT / "ignore_affected_pixels.csv",
    index=False
)

# =====================================================
# 8. FINAL REPORT
# =====================================================

print("\n5. INVESTIGATION COMPLETED")

print("Total ignore occurrences:", total_ignore)

print("\nGenerated reports:")

for file in sorted(OUTPUT.glob("ignore_*")):
    print("-", file.name)

print("\n" + "=" * 70)
print("IGNORE VALUE INVESTIGATION COMPLETED")
print("=" * 70)
