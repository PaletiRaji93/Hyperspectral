
from pathlib import Path
import numpy as np
import pandas as pd
import spectral

# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_ID = "13_G"

SAMPLE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "SWIR"
    / SAMPLE_ID
)

HDR_PATH = SAMPLE_DIR / f"{SAMPLE_ID}.hdr"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "spectral_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR / f"{SAMPLE_ID}_swir_spectral_dataset.npz"
)

print("=" * 70)
print("TERRASPECTRA - SWIR SPECTRAL EXTRACTION")
print("=" * 70)

# =====================================================
# 2. CHECK FILES
# =====================================================

if not HDR_PATH.exists():
    raise FileNotFoundError(
        f"SWIR header not found: {HDR_PATH}"
    )

for leaf_number in (1, 2):

    annotation_path = (
        SAMPLE_DIR / f"{SAMPLE_ID}_leaf{leaf_number}.csv"
    )

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation not found: {annotation_path}"
        )

print("\nSample:", SAMPLE_ID)
print("SWIR directory:", SAMPLE_DIR)

# =====================================================
# 3. LOAD SWIR HYPERSPECTRAL CUBE
# =====================================================

print("\nLoading SWIR hyperspectral cube...")

image = spectral.open_image(str(HDR_PATH))

cube = image.open_memmap()

height, width, bands = cube.shape

print("Cube shape:", cube.shape)
print("Height:", height)
print("Width:", width)
print("Spectral bands:", bands)
print("Data type:", cube.dtype)

# =====================================================
# 4. LOAD WAVELENGTHS
# =====================================================

wavelengths = np.asarray(
    image.bands.centers,
    dtype=np.float32
)

if len(wavelengths) != bands:
    raise ValueError(
        "Wavelength count does not match cube bands."
    )

print("Wavelength count:", len(wavelengths))
print("Minimum wavelength:", wavelengths.min(), "nm")
print("Maximum wavelength:", wavelengths.max(), "nm")

# =====================================================
# 5. EXTRACT LEAF SPECTRA
# =====================================================

spectral_arrays = []
label_arrays = []
coordinate_arrays = []

for leaf_number in (1, 2):

    print(f"\nExtracting Leaf {leaf_number}...")

    annotation_path = (
        SAMPLE_DIR / f"{SAMPLE_ID}_leaf{leaf_number}.csv"
    )

    coordinates = pd.read_csv(
        annotation_path,
        header=None
    ).to_numpy(dtype=np.int64)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"Invalid annotation format: {annotation_path}"
        )

    rows = coordinates[:, 0]
    cols = coordinates[:, 1]

    # Validate coordinate boundaries.
    if (
        np.any(rows < 0)
        or np.any(rows >= height)
        or np.any(cols < 0)
        or np.any(cols >= width)
    ):
        raise ValueError(
            f"Out-of-bounds coordinates in Leaf {leaf_number}"
        )

    # Extract all 256 spectral measurements per pixel.
    leaf_spectra = np.asarray(
        cube[rows, cols, :]
    )

    leaf_labels = np.full(
        len(coordinates),
        leaf_number,
        dtype=np.uint8
    )

    spectral_arrays.append(leaf_spectra)
    label_arrays.append(leaf_labels)
    coordinate_arrays.append(coordinates)

    print("Annotated pixels:", len(coordinates))
    print("Extracted spectra:", leaf_spectra.shape)
    print("Data type:", leaf_spectra.dtype)

# =====================================================
# 6. COMBINE LEAF DATA
# =====================================================

print("\nCombining extracted SWIR data...")

X = np.concatenate(
    spectral_arrays,
    axis=0
)

y = np.concatenate(
    label_arrays,
    axis=0
)

coords = np.concatenate(
    coordinate_arrays,
    axis=0
)

print("Feature matrix X:", X.shape)
print("Labels y:", y.shape)
print("Coordinates:", coords.shape)

# =====================================================
# 7. VALIDATE EXTRACTION
# =====================================================

if X.shape[1] != bands:
    raise ValueError("Incorrect spectral feature count.")

if not (len(X) == len(y) == len(coords)):
    raise ValueError("X, y and coordinates do not match.")

if not np.array_equal(np.unique(y), [1, 2]):
    raise ValueError("Expected Leaf 1 and Leaf 2 labels.")

print("\nExtraction validation:")

print("Total spectra:", len(X))
print("Leaf 1 spectra:", np.sum(y == 1))
print("Leaf 2 spectra:", np.sum(y == 2))
print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())

# =====================================================
# 8. SAVE DATASET
# =====================================================

print("\nSaving SWIR spectral dataset...")

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y,
    coords=coords,
    wavelengths=wavelengths,
    sample_id=SAMPLE_ID
)

print("Saved:", OUTPUT_PATH)

# =====================================================
# 9. FINAL REPORT
# =====================================================

print("\n" + "=" * 70)
print("SWIR SPECTRAL EXTRACTION COMPLETED")
print("=" * 70)

print("Sample ID:", SAMPLE_ID)
print("Total spectra:", X.shape[0])
print("Spectral features:", X.shape[1])
print("Leaf 1:", np.sum(y == 1))
print("Leaf 2:", np.sum(y == 2))
print("Output:", OUTPUT_PATH)

print("=" * 70)
