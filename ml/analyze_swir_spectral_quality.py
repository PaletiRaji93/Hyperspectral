
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "outputs" / "spectral_data"
    / "13_G_swir_spectral_dataset.npz"
)

OUTPUT = ROOT / "artifacts" / "swir_spectral_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

IGNORE_VALUE = 16383
MAX_UINT16 = 65535

print("=" * 75)
print("TERRASPECTRA - SWIR SPECTRAL QUALITY AUDIT")
print("=" * 75)

# =====================================================
# 2. LOAD DATA
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
wavelengths = data["wavelengths"]
coords = data["coords"]

print("\n1. DATASET INFORMATION")
print("Feature matrix:", X.shape)
print("Labels:", y.shape)
print("Coordinates:", coords.shape)
print("Wavelengths:", wavelengths.shape)
print("Data type:", X.dtype)

# =====================================================
# 3. OVERALL DATA QUALITY
# =====================================================

print("\n2. OVERALL DATA QUALITY")

print("NaN:", np.isnan(X).sum())
print("Inf:", np.isinf(X).sum())
print("Minimum:", X.min())
print("Maximum:", X.max())
print("Mean:", X.mean())
print("Standard deviation:", X.std())

total = X.size

zero_count = np.count_nonzero(X == 0)
ignore_count = np.count_nonzero(X == IGNORE_VALUE)
maximum_count = np.count_nonzero(X == MAX_UINT16)

print("\nSpecial measurement statistics:")
print("Total measurements:", total)
print("Zero measurements:", zero_count)
print("Zero percentage:", zero_count / total * 100)
print("Ignore value (16383):", ignore_count)
print("65535 measurements:", maximum_count)
print("65535 percentage:", maximum_count / total * 100)

# =====================================================
# 4. BAND-WISE STATISTICS
# =====================================================

print("\n3. BAND-WISE QUALITY")

zero_counts = np.count_nonzero(X == 0, axis=0)
ignore_counts = np.count_nonzero(X == IGNORE_VALUE, axis=0)
maximum_counts = np.count_nonzero(X == MAX_UINT16, axis=0)

band_means = X.mean(axis=0)
band_stds = X.std(axis=0)
band_mins = X.min(axis=0)
band_maxs = X.max(axis=0)

band_table = pd.DataFrame({
    "Band": np.arange(X.shape[1]),
    "Wavelength_nm": wavelengths,
    "Mean": band_means,
    "Std": band_stds,
    "Minimum": band_mins,
    "Maximum": band_maxs,
    "Zero_Count": zero_counts,
    "Zero_Percentage": zero_counts / len(X) * 100,
    "Ignore_Count": ignore_counts,
    "Ignore_Percentage": ignore_counts / len(X) * 100,
    "Maximum_Count": maximum_counts,
    "Maximum_Percentage": maximum_counts / len(X) * 100
})

band_table.to_csv(
    OUTPUT / "swir_band_quality_report.csv",
    index=False
)

print("Total bands:", X.shape[1])
print("Constant bands:", np.sum(band_stds == 0))
print("Bands containing zeros:", np.sum(zero_counts > 0))
print("Bands containing 16383:", np.sum(ignore_counts > 0))
print("Bands containing 65535:", np.sum(maximum_counts > 0))

print("\nTop 15 bands containing 65535:")

print(
    band_table.nlargest(
        15, "Maximum_Count"
    ).to_string(index=False)
)

# =====================================================
# 5. LEAF-WISE QUALITY
# =====================================================

print("\n4. LEAF-WISE QUALITY")

leaf_reports = []

for leaf in [1, 2]:

    leaf_data = X[y == leaf]

    leaf_report = {
        "Leaf_Label": leaf,
        "Pixels": len(leaf_data),
        "Zero_Count": int(np.count_nonzero(leaf_data == 0)),
        "Ignore_Count": int(np.count_nonzero(
            leaf_data == IGNORE_VALUE
        )),
        "Maximum_Count": int(np.count_nonzero(
            leaf_data == MAX_UINT16
        )),
        "Mean": float(leaf_data.mean()),
        "Std": float(leaf_data.std())
    }

    leaf_reports.append(leaf_report)

    print("\nLeaf", leaf)
    for key, value in leaf_report.items():
        if key != "Leaf_Label":
            print(f"{key}: {value}")

pd.DataFrame(leaf_reports).to_csv(
    OUTPUT / "swir_leaf_quality_report.csv",
    index=False
)

# =====================================================
# 6. PIXEL-LEVEL SPECIAL VALUE REPORT
# =====================================================

print("\n5. PIXEL-LEVEL INVESTIGATION")

zero_per_pixel = np.count_nonzero(X == 0, axis=1)
ignore_per_pixel = np.count_nonzero(
    X == IGNORE_VALUE, axis=1
)
maximum_per_pixel = np.count_nonzero(
    X == MAX_UINT16, axis=1
)

pixel_report = pd.DataFrame({
    "Row": coords[:, 0],
    "Column": coords[:, 1],
    "Leaf_Label": y,
    "Zero_Band_Count": zero_per_pixel,
    "Ignore_Band_Count": ignore_per_pixel,
    "Maximum_Value_Band_Count": maximum_per_pixel
})

pixel_report.to_csv(
    OUTPUT / "swir_pixel_quality_report.csv",
    index=False
)

print("Pixels with zero values:", np.sum(zero_per_pixel > 0))
print("Pixels with 16383:", np.sum(ignore_per_pixel > 0))
print("Pixels with 65535:", np.sum(maximum_per_pixel > 0))
print("Maximum 65535 bands in one pixel:", maximum_per_pixel.max())

# =====================================================
# 7. MEAN SPECTRAL SIGNATURES
# =====================================================

leaf1 = X[y == 1].astype(np.float64)
leaf2 = X[y == 2].astype(np.float64)

leaf1_mean = leaf1.mean(axis=0)
leaf2_mean = leaf2.mean(axis=0)

leaf1_std = leaf1.std(axis=0)
leaf2_std = leaf2.std(axis=0)

signature_table = pd.DataFrame({
    "Wavelength_nm": wavelengths,
    "Leaf1_Mean": leaf1_mean,
    "Leaf1_Std": leaf1_std,
    "Leaf2_Mean": leaf2_mean,
    "Leaf2_Std": leaf2_std,
    "Mean_Difference": leaf1_mean - leaf2_mean
})

signature_table.to_csv(
    OUTPUT / "swir_leaf_spectral_signatures.csv",
    index=False
)

plt.figure(figsize=(13, 6))
plt.plot(wavelengths, leaf1_mean, label="Leaf 1")
plt.plot(wavelengths, leaf2_mean, label="Leaf 2")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Raw Sensor Measurement")
plt.title("SWIR Mean Spectral Signatures - 13_G")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "swir_mean_spectral_signatures.png", dpi=200)
plt.close()

# =====================================================
# 8. SPECIAL VALUE DISTRIBUTION BY BAND
# =====================================================

plt.figure(figsize=(13, 5))

plt.plot(
    wavelengths,
    zero_counts / len(X) * 100,
    label="Zero measurements"
)

plt.plot(
    wavelengths,
    ignore_counts / len(X) * 100,
    label="Ignore value (16383)"
)

plt.plot(
    wavelengths,
    maximum_counts / len(X) * 100,
    label="65535 measurements"
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Measurements (%)")
plt.title("SWIR Special Value Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "swir_special_value_distribution.png", dpi=200)
plt.close()

# =====================================================
# 9. SPECTRAL VARIABILITY
# =====================================================

plt.figure(figsize=(13, 5))
plt.plot(wavelengths, band_stds)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Standard Deviation")
plt.title("SWIR Spectral Variability Across Bands")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "swir_band_standard_deviation.png", dpi=200)
plt.close()

# =====================================================
# 10. FINAL SUMMARY
# =====================================================

summary = {
    "Sample_ID": "13_G",
    "Total_Pixels": len(X),
    "Spectral_Bands": X.shape[1],
    "Total_Measurements": total,
    "Zero_Count": int(zero_count),
    "Ignore_Count": int(ignore_count),
    "Maximum_65535_Count": int(maximum_count),
    "NaN_Count": int(np.isnan(X).sum()),
    "Inf_Count": int(np.isinf(X).sum()),
    "Constant_Bands": int(np.sum(band_stds == 0))
}

pd.DataFrame([summary]).to_csv(
    OUTPUT / "swir_quality_summary.csv",
    index=False
)

print("\n6. GENERATED REPORTS")

for file in sorted(OUTPUT.iterdir()):
    print("-", file.name)

print("\n" + "=" * 75)
print("SWIR SPECTRAL QUALITY AUDIT COMPLETED")
print("=" * 75)

