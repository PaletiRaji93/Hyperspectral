
from pathlib import Path
import pandas as pd

ROOT = Path("dataset/raw/VNIR/13_G")

IMAGE_HEIGHT = 1370
IMAGE_WIDTH = 1024

files = [
    "13_G_leaf1.csv",
    "13_G_leaf2.csv"
]

print("=" * 65)
print("TERRASPECTRA - COORDINATE VALIDATION")
print("=" * 65)

for filename in files:

    path = ROOT / filename

    # CSV has no header
    df = pd.read_csv(path, header=None)

    if df.shape[1] != 2:
        raise ValueError(f"Unexpected CSV format: {filename}")

    df = df.apply(pd.to_numeric)

    rows = df.iloc[:, 0]
    cols = df.iloc[:, 1]

    valid = (
        (rows >= 0) &
        (rows < IMAGE_HEIGHT) &
        (cols >= 0) &
        (cols < IMAGE_WIDTH)
    )

    print(f"\nFile: {filename}")
    print("-" * 50)

    print("Total points:", len(df))
    print("Unique points:", len(df.drop_duplicates()))
    print("Duplicate points:", df.duplicated().sum())

    print("\nRow range:", rows.min(), "-", rows.max())
    print("Column range:", cols.min(), "-", cols.max())

    print("\nValid points:", valid.sum())
    print("Invalid points:", (~valid).sum())

    print("\nFirst 10 coordinate pairs:")
    print(df.head(10).to_string(index=False, header=False))

    if valid.all():
        print("\nSTATUS: All coordinates inside original image bounds.")
    else:
        print("\nSTATUS: Some coordinates outside image bounds.")

print("\n" + "=" * 65)
print("VALIDATION COMPLETED")
print("=" * 65)
