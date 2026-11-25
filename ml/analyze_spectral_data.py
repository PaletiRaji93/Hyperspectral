
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. PATHS
# =====================================================

DATA_PATH = Path("outputs/spectral_data/13_G_spectral_dataset.npz")

OUTPUT = Path("artifacts/spectral_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TERRASPECTRA - SPECTRAL DATA QUALITY ANALYSIS")
print("=" * 70)

# =====================================================
# 2. LOAD DATA
# =====================================================

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
wavelengths = data["wavelengths"]

print("\n1. DATASET INFORMATION")
print("Feature matrix:", X.shape)
print("Labels:", y.shape)
print("Wavelengths:", wavelengths.shape)
print("Data type:", X.dtype)

# =====================================================
# 3. CHECK DATA QUALITY
# =====================================================

print("\n2. DATA QUALITY")

print("Contains NaN:", np.isnan(X).any())
print("Contains Inf:", np.isinf(X).any())
print("Minimum:", X.min())
print("Maximum:", X.max())
print("Mean:", X.mean())
print("Standard deviation:", X.std())

# =====================================================
# 4. ZERO VALUES PER BAND
# =====================================================

zero_counts = np.count_nonzero(X == 0, axis=0)

zero_percentages = (
    zero_counts / X.shape[0]
) * 100

# =====================================================
# 5. BAND STATISTICS
# =====================================================

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
    "Zero_Percentage": zero_percentages
})

band_table.to_csv(
    OUTPUT / "band_quality_report.csv",
    index=False
)

print("\n3. BAND QUALITY")
print("Total bands:", X.shape[1])
print("Constant bands:", np.sum(band_stds == 0))
print("Bands containing zero values:", np.sum(zero_counts > 0))

print("\nBands with highest zero percentages:")
print(
    band_table.nlargest(
        10, "Zero_Percentage"
    ).to_string(index=False)
)

# =====================================================
# 6. LEAF-WISE SPECTRAL STATISTICS
# =====================================================

leaf1 = X[y == 1]
leaf2 = X[y == 2]

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

print("\n4. LEAF COMPARISON")
print("Leaf 1 mean intensity:", leaf1.mean())
print("Leaf 2 mean intensity:", leaf2.mean())

# =====================================================
# 7. PLOT MEAN SPECTRAL SIGNATURES
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
plt.title("Mean Spectral Signatures - 13_G")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "mean_spectral_signatures.png",
    dpi=200
)

plt.close()

# =====================================================
# 8. ZERO PERCENTAGE BY BAND
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(
    wavelengths,
    zero_percentages,
    linewidth=1.5
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Zero Values (%)")
plt.title("Zero Measurement Percentage Across Bands")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "zero_values_per_band.png",
    dpi=200
)

plt.close()

# =====================================================
# 9. BAND STANDARD DEVIATION
# =====================================================

plt.figure(figsize=(12, 5))

plt.plot(
    wavelengths,
    band_stds,
    linewidth=1.5
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Standard Deviation")
plt.title("Spectral Variability Across Bands")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "band_standard_deviation.png",
    dpi=200
)

plt.close()

# =====================================================
# 10. FINAL REPORT
# =====================================================

print("\n5. ANALYSIS OUTPUTS")

for file in sorted(OUTPUT.iterdir()):
    print("-", file.name)

print("\n" + "=" * 70)
print("SPECTRAL QUALITY ANALYSIS COMPLETED")
print("=" * 70)
