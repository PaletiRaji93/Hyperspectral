
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

VNIR_PATH = ROOT / "outputs/spectral_data/13_G_spectral_dataset.npz"
SWIR_PATH = ROOT / "outputs/spectral_data/13_G_swir_spectral_dataset.npz"

OUTPUT_DIR = ROOT / "artifacts/multisensor_preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path, sensor):
    if not path.exists():
        raise FileNotFoundError(f"{sensor} dataset not found: {path}")

    data = np.load(path)

    required = ["X", "y", "coords", "wavelengths"]
    missing = [key for key in required if key not in data]

    if missing:
        raise ValueError(f"{sensor} missing fields: {missing}")

    X = data["X"]
    y = data["y"]
    coords = data["coords"]
    wavelengths = data["wavelengths"]

    print(f"\n{sensor} DATASET")
    print("-" * 55)
    print("Features:", X.shape)
    print("Labels:", y.shape)
    print("Coordinates:", coords.shape)
    print("Wavelengths:", wavelengths.shape)
    print("Data type:", X.dtype)
    print("Label values:", np.unique(y))
    print("NaN:", np.isnan(X).sum())
    print("Inf:", np.isinf(X).sum())

    if X.ndim != 2:
        raise ValueError(f"{sensor}: X must be a 2D matrix")

    if X.shape[0] != len(y) or X.shape[0] != len(coords):
        raise ValueError(f"{sensor}: inconsistent sample counts")

    if X.shape[1] != len(wavelengths):
        raise ValueError(f"{sensor}: wavelength count mismatch")

    if len(np.unique(y)) < 2:
        raise ValueError(f"{sensor}: expected at least two leaf labels")

    return {
        "X": X,
        "y": y,
        "coords": coords,
        "wavelengths": wavelengths
    }


def main():
    print("=" * 75)
    print("TERRASPECTRA - MULTISENSOR DATASET PREPARATION AUDIT")
    print("=" * 75)

    vnir = load_dataset(VNIR_PATH, "VNIR")
    swir = load_dataset(SWIR_PATH, "SWIR")

    print("\nSENSOR COMPARISON")
    print("-" * 55)

    print("VNIR pixels:", len(vnir["y"]))
    print("SWIR pixels:", len(swir["y"]))
    print("VNIR bands:", len(vnir["wavelengths"]))
    print("SWIR bands:", len(swir["wavelengths"]))

    print("\nIMPORTANT VALIDATION")
    print("-" * 55)

    print("Same number of pixels:",
          len(vnir["y"]) == len(swir["y"]))

    print("Same label counts:",
          np.array_equal(
              np.unique(vnir["y"], return_counts=True)[1],
              np.unique(swir["y"], return_counts=True)[1]
          ))

    print("VNIR wavelength range:",
          round(float(vnir["wavelengths"].min()), 2),
          "-",
          round(float(vnir["wavelengths"].max()), 2), "nm")

    print("SWIR wavelength range:",
          round(float(swir["wavelengths"].min()), 2),
          "-",
          round(float(swir["wavelengths"].max()), 2), "nm")

    overlap_min = max(
        vnir["wavelengths"].min(),
        swir["wavelengths"].min()
    )

    overlap_max = min(
        vnir["wavelengths"].max(),
        swir["wavelengths"].max()
    )

    print("Spectral overlap:",
          round(float(overlap_min), 2),
          "-",
          round(float(overlap_max), 2), "nm")

    report = pd.DataFrame([
        {
            "Sensor": "VNIR",
            "Pixels": len(vnir["y"]),
            "Bands": len(vnir["wavelengths"]),
            "Wavelength_Min_nm": vnir["wavelengths"].min(),
            "Wavelength_Max_nm": vnir["wavelengths"].max(),
            "Data_Type": str(vnir["X"].dtype)
        },
        {
            "Sensor": "SWIR",
            "Pixels": len(swir["y"]),
            "Bands": len(swir["wavelengths"]),
            "Wavelength_Min_nm": swir["wavelengths"].min(),
            "Wavelength_Max_nm": swir["wavelengths"].max(),
            "Data_Type": str(swir["X"].dtype)
        }
    ])

    report_path = OUTPUT_DIR / "multisensor_dataset_report.csv"
    report.to_csv(report_path, index=False)

    print("\nREPORT SAVED:")
    print(report_path)

    print("\nNo original data was modified.")
    print("=" * 75)
    print("MULTISENSOR AUDIT COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
