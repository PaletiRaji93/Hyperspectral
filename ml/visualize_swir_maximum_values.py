
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "outputs" / "spectral_data"
    / "13_G_swir_spectral_dataset.npz"
)

OUTPUT = ROOT / "outputs" / "swir_visualization"
OUTPUT.mkdir(parents=True, exist_ok=True)

MAX_VALUE = 65535

print("=" * 70)
print("TERRASPECTRA - SWIR MAXIMUM VALUE SPATIAL ANALYSIS")
print("=" * 70)

# =====================================================
# 2. LOAD DATA
# =====================================================

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
coords = data["coords"]
wavelengths = data["wavelengths"]

print("\nDataset shape:", X.shape)

# =====================================================
# 3. IDENTIFY MAXIMUM-VALUE MEASUREMENTS
# =====================================================

affected_rows, affected_bands = np.where(X == MAX_VALUE)

affected_coords = coords[affected_rows]
affected_labels = y[affected_rows]

affected_wavelengths = wavelengths[affected_bands]

print("\nMaximum-value statistics")
print("Total occurrences:", len(affected_rows))
print("Affected pixels:", len(np.unique(affected_rows)))
print("Affected bands:", len(np.unique(affected_bands)))

# =====================================================
# 4. CREATE DETAILED CSV REPORT
# =====================================================

report = pd.DataFrame({
    "Row": affected_coords[:, 0],
    "Column": affected_coords[:, 1],
    "Leaf_Label": affected_labels,
    "Band": affected_bands,
    "Wavelength_nm": affected_wavelengths,
    "Measurement": MAX_VALUE
})

report.to_csv(
    OUTPUT / "swir_maximum_value_locations.csv",
    index=False
)

print("\nOccurrences by band:")

print(
    report.groupby(
        ["Band", "Wavelength_nm"]
    ).size().reset_index(
        name="Occurrence_Count"
    ).to_string(index=False)
)

print("\nOccurrences by leaf:")
print(report["Leaf_Label"].value_counts().sort_index())

# =====================================================
# 5. SPATIAL VISUALIZATION
# =====================================================

plt.figure(figsize=(12, 8))

for leaf in [1, 2]:

    leaf_coords = coords[y == leaf]

    plt.scatter(
        leaf_coords[:, 1],
        leaf_coords[:, 0],
        s=2,
        alpha=0.15,
        label=f"Leaf {leaf} pixels"
    )

plt.scatter(
    affected_coords[:, 1],
    affected_coords[:, 0],
    c=affected_bands,
    cmap="turbo",
    s=20,
    edgecolors="black",
    linewidths=0.3,
    label="65535 measurements"
)

plt.colorbar(label="Affected spectral band")

plt.xlabel("Column")
plt.ylabel("Row")
plt.title("SWIR 65535 Spatial Distribution - 13_G")

plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    OUTPUT / "swir_maximum_value_spatial_distribution.png",
    dpi=200
)

plt.close()

# =====================================================
# 6. LEAF-WISE VISUALIZATION
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, leaf in zip(axes, [1, 2]):

    mask = affected_labels == leaf

    leaf_coords = coords[y == leaf]

    ax.scatter(
        leaf_coords[:, 1],
        leaf_coords[:, 0],
        s=2,
        alpha=0.2
    )

    ax.scatter(
        affected_coords[mask, 1],
        affected_coords[mask, 0],
        c=affected_bands[mask],
        cmap="turbo",
        s=25,
        edgecolors="black",
        linewidths=0.3
    )

    ax.set_title(f"Leaf {leaf} - 65535 locations")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.invert_yaxis()

plt.tight_layout()

plt.savefig(
    OUTPUT / "swir_maximum_value_leaf_distribution.png",
    dpi=200
)

plt.close()

# =====================================================
# 7. FINAL REPORT
# =====================================================

print("\nGenerated files:")

for file in sorted(OUTPUT.iterdir()):
    print("-", file.name)

print("\n" + "=" * 70)
print("SWIR SPATIAL ANALYSIS COMPLETED")
print("=" * 70)
