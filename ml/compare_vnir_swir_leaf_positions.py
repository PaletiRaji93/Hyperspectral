
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectral import envi

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = "13_G"

VNIR_DIR = ROOT / "dataset" / "raw" / "VNIR"
SWIR_DIR = ROOT / "dataset" / "raw" / "SWIR"

def load_coordinates(folder, leaf):
    path = folder / SAMPLE / f"{SAMPLE}_{leaf}.csv"
    return pd.read_csv(path, header=None).iloc[:, :2].to_numpy(dtype=int)

def get_bounds(coords):
    return {
        "min_row": coords[:, 0].min(),
        "max_row": coords[:, 0].max(),
        "min_col": coords[:, 1].min(),
        "max_col": coords[:, 1].max(),
        "center_row": coords[:, 0].mean(),
        "center_col": coords[:, 1].mean(),
        "pixel_count": len(coords)
    }

records = []

for sensor, folder in [("VNIR", VNIR_DIR), ("SWIR", SWIR_DIR)]:
    print(f"\n{sensor} LEAF POSITION ANALYSIS")

    for leaf in ["leaf1", "leaf2"]:
        coords = load_coordinates(folder, leaf)
        bounds = get_bounds(coords)

        records.append({
            "sensor": sensor,
            "leaf": leaf,
            **bounds
        })

        print(f"\n{leaf}")
        for key, value in bounds.items():
            print(f"{key}: {value}")

df = pd.DataFrame(records)

output_dir = ROOT / "artifacts" / "dataset_inspection"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{SAMPLE}_vnir_swir_leaf_positions.csv"
df.to_csv(output_file, index=False)

print("\nComparison report saved:")
print(output_file)

print("\nPOSITION COMPARISON")
print(df.to_string(index=False))
