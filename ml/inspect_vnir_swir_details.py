
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SENSORS = {
    "VNIR": ROOT / "dataset" / "raw" / "VNIR",
    "SWIR": ROOT / "dataset" / "raw" / "SWIR",
}

SAMPLE = "13_G"


def read_header(path):
    text = path.read_text(errors="ignore")

    def get(pattern):
        match = re.search(pattern, text, re.I | re.M)
        return match.group(1).strip() if match else "Not found"

    wavelengths = re.search(
        r"wavelength\s*=\s*\{([^}]*)\}", text, re.I | re.S
    )

    if wavelengths:
        values = np.fromstring(
            wavelengths.group(1).replace(",", " "),
            sep=" "
        )
    else:
        values = np.array([])

    return {
        "lines": get(r"^\s*lines\s*=\s*(\d+)"),
        "samples": get(r"^\s*samples\s*=\s*(\d+)"),
        "bands": get(r"^\s*bands\s*=\s*(\d+)"),
        "interleave": get(r"^\s*interleave\s*=\s*(\S+)"),
        "data_type": get(r"^\s*data type\s*=\s*(\d+)"),
        "header_offset": get(r"^\s*header offset\s*=\s*(\d+)"),
        "ignore_value": get(r"^\s*data ignore value\s*=\s*([^\r\n]+)"),
        "wavelength_units": get(
            r"^\s*wavelength units\s*=\s*([^\r\n]+)"
        ),
        "wavelengths": values,
    }


for sensor, folder in SENSORS.items():

    print("\n" + "=" * 65)
    print(sensor, "DETAILED INSPECTION")
    print("=" * 65)

    sample_folder = next(
        (p for p in folder.rglob(SAMPLE + ".hdr")),
        None
    )

    if sample_folder is None:
        print("Sample header not found")
        continue

    info = read_header(sample_folder)

    for key, value in info.items():
        if key != "wavelengths":
            print(f"{key}: {value}")

    wavelengths = info["wavelengths"]

    print("Wavelength count:", len(wavelengths))

    if len(wavelengths):
        print("Minimum wavelength:", wavelengths.min())
        print("Maximum wavelength:", wavelengths.max())
        print("First 10 wavelengths:", wavelengths[:10])
        print("Last 10 wavelengths:", wavelengths[-10:])

    # Inspect annotation coordinate files
    print("\nANNOTATION COORDINATES")

    annotation_files = sorted(
        folder.rglob(SAMPLE + "_leaf*.csv")
    )

    if not annotation_files:
        print("No annotation CSV files found for", SAMPLE)

    for csv_path in annotation_files:

        df = pd.read_csv(csv_path, header=None)

        print("\nFile:", csv_path.name)
        print("Shape:", df.shape)

        if df.shape[1] < 2:
            print("Unexpected coordinate format")
            continue

        coords = df.iloc[:, :2].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if coords.empty:
            print("No valid coordinates")
            continue

        rows = coords.iloc[:, 0]
        cols = coords.iloc[:, 1]

        height = int(info["lines"])
        width = int(info["samples"])

        valid = (
            (rows >= 0) & (rows < height) &
            (cols >= 0) & (cols < width)
        )

        print("Coordinate count:", len(coords))
        print("Row range:", rows.min(), "-", rows.max())
        print("Column range:", cols.min(), "-", cols.max())
        print("Valid coordinates:", valid.sum())
        print("Invalid coordinates:", (~valid).sum())

print("\nDETAILED INSPECTION COMPLETED")
