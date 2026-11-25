
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

# =====================================================
# 2. LOAD DATA
# =====================================================

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
wavelengths = data["wavelengths"]

print("=" * 70)
print("TERRASPECTRA - ZERO MEASUREMENT INVESTIGATION")
print("=" * 70)

print("\nDataset shape:", X.shape)

# =====================================================
# 3. LOAD ORIGINAL ENVI METADATA
# =====================================================

image = open_image(str(HDR_PATH))
metadata = image.metadata

print("\n1. ENVI METADATA")

print("Available metadata fields:")

for key in metadata:
    print("-", key)

print("\nImportant metadata values:")

keywords = [
    "wavelength",
    "bbl",
    "ignore",
    "reflectance",
    "scale",
    "dark",
    "white",
    "gain",
    "offset",
    "interleave",
    "sensor",
    "data type",
    "description"
]

for key, value in metadata.items():

    if any(word in key.lower() for word in keywords):

        if key.lower() == "wavelength":
            continue

        print(f"\n{key}: {value}")

# =====================================================
# 4. ZERO MEASUREMENTS PER LEAF
# =====================================================

print("\n2. ZERO MEASUREMENTS PER LEAF")

leaf_results = []

for label in [1, 2]:

    leaf = X[y == label]

    zero_count = np.count_nonzero(leaf == 0)
    total_values = leaf.size

    zero_percentage = (
        zero_count / total_values
    ) * 100

    affected_pixels = np.any(leaf == 0, axis=1).sum()

    print(f"\nLeaf {label}")
    print("Shape:", leaf.shape)
    print("Zero measurements:", zero_count)
    print("Zero percentage:", zero_percentage)
    print("Pixels containing at least one zero:", affected_pixels)
    print("Affected pixel percentage:",
          affected_pixels / len(leaf) * 100)

    leaf_results.append({
        "Leaf": label,
        "Total_Pixels": len(leaf),
        "Zero_Count": zero_count,
        "Zero_Percentage": zero_percentage,
        "Affected_Pixels": affected_pixels
    })

pd.DataFrame(leaf_results).to_csv(
    OUTPUT / "leaf_zero_measurement_report.csv",
    index=False
)

# =====================================================
# 5. ZERO DISTRIBUTION PER BAND
# =====================================================

print("\n3. ZERO DISTRIBUTION PER BAND")

zero_counts = np.count_nonzero(X == 0, axis=0)

zero_percentages = zero_counts / len(X) * 100

leaf1 = X[y == 1]
leaf2 = X[y == 2]

band_report = pd.DataFrame({
    "Band": np.arange(X.shape[1]),
    "Wavelength_nm": wavelengths,
    "Leaf1_Zero_Percentage":
        np.mean(leaf1 == 0, axis=0) * 100,
    "Leaf2_Zero_Percentage":
        np.mean(leaf2 == 0, axis=0) * 100,
    "Combined_Zero_Count": zero_counts,
    "Combined_Zero_Percentage": zero_percentages
})

print("\nTop 15 bands by zero percentage:")

print(
    band_report.nlargest(
        15,
        "Combined_Zero_Percentage"
    ).to_string(index=False)
)

band_report.to_csv(
    OUTPUT / "detailed_zero_band_report.csv",
    index=False
)

# =====================================================
# 6. CHECK PIXELS WITH MANY ZERO VALUES
# =====================================================

print("\n4. PIXEL-LEVEL ZERO ANALYSIS")

zeros_per_pixel = np.sum(X == 0, axis=1)

print("Maximum zeros in one pixel:", zeros_per_pixel.max())
print("Mean zeros per pixel:", zeros_per_pixel.mean())

for threshold in [1, 5, 10, 20, 50]:

    affected = np.sum(zeros_per_pixel >= threshold)

    print(
        f"Pixels with at least {threshold} zero bands:",
        affected
    )

# =====================================================
# 7. SPECTRAL PERCENTILES
# =====================================================

print("\n5. SPECTRAL VALUE DISTRIBUTION")

percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]

percentile_values = np.percentile(
    X,
    percentiles,
    axis=0
)

percentile_report = pd.DataFrame(
    percentile_values.T,
    columns=[f"P{p}" for p in percentiles]
)

percentile_report.insert(
    0,
    "Wavelength_nm",
    wavelengths
)

percentile_report.to_csv(
    OUTPUT / "spectral_percentile_report.csv",
    index=False
)

# =====================================================
# 8. CHECK STORAGE-LEVEL MAXIMUM VALUES
# =====================================================

print("\n6. EXTREME VALUE CHECK")

print("Maximum dataset value:", X.max())
print("uint16 storage maximum:", np.iinfo(np.uint16).max)

print(
    "Values equal to uint16 maximum:",
    np.sum(X == np.iinfo(np.uint16).max)
)

# =====================================================
# 9. FINAL SUMMARY
# =====================================================

print("\n7. OUTPUT FILES")

for file in sorted(OUTPUT.iterdir()):
    print("-", file.name)

print("\n" + "=" * 70)
print("ZERO MEASUREMENT INVESTIGATION COMPLETED")
print("=" * 70)
