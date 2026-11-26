
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "spectral_data"
    / "13_G_spectral_dataset.npz"
)

OUTPUT = PROJECT_ROOT / "artifacts" / "spectral_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

IGNORE_VALUE = 4095

print("=" * 75)
print("TERRASPECTRA - VNIR SPECTRAL QUALITY AUDIT")
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

print("Contains NaN:", np.isnan(X).any())
print("Contains Inf:", np.isinf(X).any())
print("Minimum:", X.min())
print("Maximum:", X.max())
print("Mean:", X.mean())
print("Standard deviation:", X.std())

print("Total measurements:", X.size)
print("Zero measurements:", np.count_nonzero(X == 0))
print("Ignore-value measurements:", np.count_nonzero(X == IGNORE_VALUE))
print("Maximum uint16 measurements:", np.count_nonzero(X == 65535))

# =====================================================
# 4. BAND-WISE QUALITY STATISTICS
# =====================================================

print("\n3. BAND-WISE QUALITY ANALYSIS")

zero_counts = np.count_nonzero(X == 0, axis=0)

ignore_counts = np.count_nonzero(
    X == IGNORE_VALUE, axis=0
)

maximum_counts = np.count_nonzero(
    X == 65535, axis=0
)

band_means = X.mean(axis=0)
band_stds = X.std(axis=0)
band_mins = X.min(axis=0)
band_maxs = X.max(axis=0)

zero_percentages = zero_counts / len(X) * 100
ignore_percentages = ignore_counts / len(X) * 100
maximum_percentages = maximum_counts / len(X) * 100

band_table = pd.DataFrame({
    "Band": np.arange(X.shape[1]),
    "Wavelength_nm": wavelengths,
    "Mean": band_means,
    "Std": band_stds,
    "Minimum": band_mins,
    "Maximum": band_maxs,
    "Zero_Count": zero_counts,
    "Zero_Percentage": zero_percentages,
    "Ignore_Count": ignore_counts,
    "Ignore_Percentage": ignore_percentages,
    "Maximum_Value_Count": maximum_counts,
    "Maximum_Value_Percentage": maximum_percentages
})

band_table.to_csv(
    OUTPUT / "band_quality_report.csv",
    index=False
)

print("Total bands:", X.shape[1])
print("Constant bands:", np.sum(band_stds == 0))
print("Bands containing zeros:", np.sum(zero_counts > 0))
print("Bands containing ignore values:", np.sum(ignore_counts > 0))
print("Bands containing 65535:", np.sum(maximum_counts > 0))

print("\nTop 10 bands by zero percentage:")

print(
    band_table.nlargest(
        10, "Zero_Percentage"
    ).to_string(index=False)
)

print("\nBands containing ignore values:")

print(
    band_table[band_table["Ignore_Count"] > 0].to_string(index=False)
)

# =====================================================
# 5. LEAF-WISE SPECTRAL STATISTICS
# =====================================================

print("\n4. LEAF-WISE SPECTRAL ANALYSIS")

leaf1 = X[y == 1].astype(np.float64)
leaf2 = X[y == 2].astype(np.float64)

leaf1_mean = leaf1.mean(axis=0)
leaf2_mean = leaf2.mean(axis=0)

leaf1_std = leaf1.std(axis=0)
leaf2_std = leaf2.std(axis=0)

leaf_summary = pd.DataFrame({
    "Wavelength_nm": wavelengths,
    "Leaf1_Mean": leaf1_mean,
    "Leaf1_Std": leaf1_std,
    "Leaf2_Mean": leaf2_mean,
    "Leaf2_Std": leaf2_std,
    "Mean_Difference": leaf1_mean - leaf2_mean
})

leaf_summary.to_csv(
    OUTPUT / "leaf_spectral_comparison.csv",
    index=False
)

print("Leaf 1 pixels:", len(leaf1))
print("Leaf 2 pixels:", len(leaf2))
print("Leaf 1 mean intensity:", leaf1.mean())
print("Leaf 2 mean intensity:", leaf2.mean())

# =====================================================
# 6. PLOT MEAN SPECTRAL SIGNATURES
# =====================================================

plt.figure(figsize=(12, 6))

plt.plot(
    wavelengths,
    leaf1_mean,
    label="Leaf 1",
    linewidth=2
)

plt.plot(
    wavelengths,
    leaf2_mean,
    label="Leaf 2",
    linewidth=2
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Mean Raw Sensor Measurement")
plt.title("VNIR Mean Spectral Signatures - 13_G")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "mean_spectral_signatures.png",
    dpi=200
)

plt.close()

# =====================================================
# 7. ZERO AND IGNORE VALUE DISTRIBUTION
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(
    wavelengths,
    zero_percentages,
    label="Zero measurements"
)

plt.plot(
    wavelengths,
    ignore_percentages,
    label="Ignore value (4095)"
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Measurements (%)")
plt.title("VNIR Invalid/Unusual Measurement Distribution")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "zero_and_ignore_distribution.png",
    dpi=200
)

plt.close()

# =====================================================
# 8. BAND STANDARD DEVIATION
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(wavelengths, band_stds)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Standard Deviation")
plt.title("VNIR Spectral Variability Across Bands")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "band_standard_deviation.png",
    dpi=200
)

plt.close()

# =====================================================
# 9. FINAL REPORT
# =====================================================

summary = {
    "Sample_ID": "13_G",
    "Total_Spectra": len(X),
    "Spectral_Bands": X.shape[1],
    "NaN_Count": int(np.isnan(X).sum()),
    "Inf_Count": int(np.isinf(X).sum()),
    "Zero_Count": int(np.count_nonzero(X == 0)),
    "Ignore_Value_Count": int(np.count_nonzero(X == IGNORE_VALUE)),
    "Maximum_Value_Count": int(np.count_nonzero(X == 65535)),
    "Constant_Bands": int(np.sum(band_stds == 0))
}

pd.DataFrame([summary]).to_csv(
    OUTPUT / "spectral_quality_summary.csv",
    index=False
)

print("\n5. GENERATED REPORTS")

for file in sorted(OUTPUT.iterdir()):
    print("-", file.name)

print("\n" + "=" * 75)
print("VNIR SPECTRAL QUALITY AUDIT COMPLETED")
print("=" * 75)
