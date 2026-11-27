
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "VNIR": ROOT / "outputs/spectral_data/13_G_spectral_dataset.npz",
    "SWIR": ROOT / "outputs/spectral_data/13_G_swir_spectral_dataset.npz"
}

METADATA_PATH = ROOT / "artifacts/dataset_inspection/master_sample_metadata.csv"

OUTPUT_DIR = ROOT / "outputs/leaf_level_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_features(sensor, dataset_path):
    print(f"\n{'=' * 65}")
    print(f"PROCESSING {sensor}")
    print("=" * 65)

    data = np.load(dataset_path)

    X = data["X"].astype(np.float32)
    y = data["y"]
    wavelengths = data["wavelengths"]

    rows = []

    for leaf_label in np.unique(y):
        leaf_data = X[y == leaf_label]

        if len(leaf_data) == 0:
            continue

        print(f"\nLeaf annotation: {leaf_label}")
        print("Pixel count:", len(leaf_data))

        # Calculate one spectral feature vector per leaf.
        mean_values = np.mean(leaf_data, axis=0)
        std_values = np.std(leaf_data, axis=0)
        median_values = np.median(leaf_data, axis=0)
        q25_values = np.percentile(leaf_data, 25, axis=0)
        q75_values = np.percentile(leaf_data, 75, axis=0)

        feature_row = {
            "Sample_ID": "13_G",
            "Leaf_Label": int(leaf_label),
            "Pixel_Count": len(leaf_data),
            "Sensor": sensor
        }

        for i, wavelength in enumerate(wavelengths):
            band_name = f"{wavelength:.2f}nm"

            feature_row[f"mean_{band_name}"] = mean_values[i]
            feature_row[f"std_{band_name}"] = std_values[i]
            feature_row[f"median_{band_name}"] = median_values[i]
            feature_row[f"q25_{band_name}"] = q25_values[i]
            feature_row[f"q75_{band_name}"] = q75_values[i]

        # Overall statistics for the leaf.
        feature_row["Overall_Mean"] = np.mean(leaf_data)
        feature_row["Overall_Std"] = np.std(leaf_data)
        feature_row["Overall_Min"] = np.min(leaf_data)
        feature_row["Overall_Max"] = np.max(leaf_data)

        rows.append(feature_row)

    result = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / f"13_G_{sensor.lower()}_leaf_features.csv"
    result.to_csv(output_path, index=False)

    print("\nGenerated feature matrix:", result.shape)
    print("Saved:", output_path)

    return result


def inspect_metadata():
    print(f"\n{'=' * 65}")
    print("MASTER METADATA INSPECTION")
    print("=" * 65)

    if not METADATA_PATH.exists():
        print("Master metadata file not found:")
        print(METADATA_PATH)
        return

    metadata = pd.read_csv(METADATA_PATH)

    print("\nMetadata shape:", metadata.shape)
    print("\nColumn names:")

    for column in metadata.columns:
        print("-", column)

    # Find the sample ID column without assuming capitalization.
    sample_column = next(
        (
            column for column in metadata.columns
            if column.strip().lower() in
            ["sample_id", "sampleid", "sample"]
        ),
        None
    )

    if sample_column is None:
        print("\nCould not automatically identify sample ID column.")
        print("Please share the printed column names.")
        return

    sample_rows = metadata[
        metadata[sample_column].astype(str).str.strip() == "13_G"
    ]

    print("\nMetadata records for sample 13_G:")
    print("Number of records:", len(sample_rows))

    if len(sample_rows):
        print(sample_rows.to_string(index=False))


def main():
    print("=" * 75)
    print("TERRASPECTRA - LEAF LEVEL FEATURE ENGINEERING")
    print("=" * 75)

    for sensor, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {sensor} dataset: {path}")

    vnir_features = build_features("VNIR", DATASETS["VNIR"])
    swir_features = build_features("SWIR", DATASETS["SWIR"])

    print("\nFEATURE SUMMARY")
    print("-" * 60)
    print("VNIR leaf records:", len(vnir_features))
    print("SWIR leaf records:", len(swir_features))
    print("VNIR feature columns:", len(vnir_features.columns))
    print("SWIR feature columns:", len(swir_features.columns))

    inspect_metadata()

    print("\nOriginal spectral datasets were not modified.")
    print("\nLEAF LEVEL FEATURE ENGINEERING COMPLETED")


if __name__ == "__main__":
    main()
