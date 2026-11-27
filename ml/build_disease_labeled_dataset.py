
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "outputs/leaf_level_features"
METADATA_PATH = ROOT / "artifacts/dataset_inspection/master_sample_metadata.csv"

OUTPUT_DIR = ROOT / "outputs/disease_labeled_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_sensor_dataset(sensor):
    print(f"\n{'=' * 65}")
    print(f"BUILDING {sensor} DISEASE-LABELLED DATASET")
    print("=" * 65)

    feature_path = FEATURE_DIR / f"13_g_{sensor.lower()}_leaf_features.csv"

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    features = pd.read_csv(feature_path)
    metadata = pd.read_csv(METADATA_PATH)

    # Keep only records eligible for leaf-level analysis.
    metadata = metadata[
        metadata["Eligible_For_Leaf_Analysis"].astype(str).str.lower()
        == "true"
    ].copy()

    # Select the correct annotation availability field.
    annotation_column = f"{sensor}_Leaf_Annotation"

    metadata = metadata[
        metadata[annotation_column].astype(str).str.lower() == "true"
    ].copy()

    # Join using sample ID and leaf index.
    merged = features.merge(
        metadata,
        left_on=["Sample_ID", "Leaf_Label"],
        right_on=["Sample_ID", "Leaf"],
        how="inner",
        validate="one_to_one"
    )

    if merged.empty:
        raise ValueError(f"No matching {sensor} records after joining.")

    # Encode treatment labels.
    treatment_mapping = {
        "Non-infected": 0,
        "Infected": 1
    }

    merged["Disease_Label"] = merged["Treatment"].map(treatment_mapping)

    if merged["Disease_Label"].isna().any():
        raise ValueError("Unknown or missing treatment labels detected.")

    merged["Disease_Label"] = merged["Disease_Label"].astype(int)

    print("\nJoined records:", len(merged))
    print("\nTreatment distribution:")
    print(merged["Treatment"].value_counts())

    print("\nDisease label distribution:")
    print(merged["Disease_Label"].value_counts().sort_index())

    print("\nGenotype distribution:")
    print(merged["Genotype"].value_counts())

    output_path = OUTPUT_DIR / f"{sensor.lower()}_disease_labeled.csv"

    merged.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Final dataset shape:", merged.shape)

    return merged


def main():
    print("=" * 75)
    print("TERRASPECTRA - DISEASE LABELLED DATASET BUILDER")
    print("=" * 75)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    vnir = prepare_sensor_dataset("VNIR")
    swir = prepare_sensor_dataset("SWIR")

    print("\nFINAL SUMMARY")
    print("-" * 60)
    print("VNIR records:", len(vnir))
    print("SWIR records:", len(swir))

    print("\nIMPORTANT:")
    print("These are leaf-level records, not independent plant samples.")
    print("Do not train or evaluate a disease classifier on this")
    print("single sample.")

    print("\nDISEASE-LABELLED DATASET BUILD COMPLETED")


if __name__ == "__main__":
    main()
