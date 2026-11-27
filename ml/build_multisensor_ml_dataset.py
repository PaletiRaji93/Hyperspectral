
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "outputs/leaf_level_features"
OUTPUT_DIR = ROOT / "outputs/ml_datasets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VNIR_PATH = FEATURE_DIR / "vnir_all_leaf_features.csv"
SWIR_PATH = FEATURE_DIR / "swir_all_leaf_features.csv"

KEYS = ["Sample_ID", "Leaf"]

METADATA_COLUMNS = [
    "Sample_ID",
    "Pot",
    "Side",
    "Leaf",
    "Acquisition_Date",
    "DPI",
    "DPI_Mapping_Status",
    "Genotype",
    "Treatment",
    "Disease_Label",
    "Confidence",
    "Pixel_Count"
]

FEATURE_PREFIXES = (
    "mean_",
    "std_",
    "median_",
    "q25_",
    "q75_"
)


def load_features(path, sensor):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if df.duplicated(KEYS).any():
        raise ValueError(f"{sensor}: duplicate sample/leaf records")

    feature_columns = [
        col for col in df.columns
        if col.startswith(FEATURE_PREFIXES)
    ]

    if not feature_columns:
        raise ValueError(f"{sensor}: no spectral feature columns found")

    if df[feature_columns].isna().any().any():
        missing_count = int(df[feature_columns].isna().sum().sum())
        raise ValueError(
            f"{sensor}: {missing_count} missing spectral feature values. "
            "Investigate before model training."
        )

    print(f"\n{sensor} DATASET")
    print("-" * 55)
    print("Records:", len(df))
    print("Unique samples:", df["Sample_ID"].nunique())
    print("Spectral feature columns:", len(feature_columns))
    print("Disease distribution:")
    print(df["Treatment"].value_counts().to_string())

    return df, feature_columns


def check_metadata_match(vnir, swir):
    print("\nCROSS-SENSOR VALIDATION")
    print("-" * 55)

    if len(vnir) != len(swir):
        raise ValueError("VNIR and SWIR record counts differ")

    vnir_keys = set(map(tuple, vnir[KEYS].to_numpy()))
    swir_keys = set(map(tuple, swir[KEYS].to_numpy()))

    if vnir_keys != swir_keys:
        raise ValueError("VNIR and SWIR sample/leaf records do not match")

    print("Matching sample/leaf records:", len(vnir_keys))

    comparison_columns = [
        "Pot",
        "Side",
        "Treatment",
        "Disease_Label",
        "Genotype",
        "Confidence"
    ]

    left = vnir[KEYS + comparison_columns].copy()
    right = swir[KEYS + comparison_columns].copy()

    comparison = left.merge(
        right,
        on=KEYS,
        suffixes=("_VNIR", "_SWIR"),
        validate="one_to_one"
    )

    for column in comparison_columns:
        left_values = comparison[f"{column}_VNIR"].fillna("<NA>").astype(str)
        right_values = comparison[f"{column}_SWIR"].fillna("<NA>").astype(str)

        if not left_values.equals(right_values):
            raise ValueError(
                f"Metadata mismatch between sensors: {column}"
            )

    print("Treatment labels match: YES")
    print("Genotypes match: YES")
    print("Confidence values match: YES")
    print("Pot and side information match: YES")


def build_sensor_table(df, feature_columns, sensor):
    columns = METADATA_COLUMNS + feature_columns

    result = df[columns].copy()

    output_path = OUTPUT_DIR / f"{sensor.lower()}_ml_dataset.csv"
    result.to_csv(output_path, index=False)

    print(f"\nSaved {sensor} dataset:")
    print(output_path)
    print("Shape:", result.shape)

    return result


def main():
    print("=" * 75)
    print("TERRASPECTRA - MULTISENSOR ML DATASET BUILDER")
    print("=" * 75)

    vnir, vnir_features = load_features(VNIR_PATH, "VNIR")
    swir, swir_features = load_features(SWIR_PATH, "SWIR")

    check_metadata_match(vnir, swir)

    # Create separate sensor-specific datasets.
    vnir_ml = build_sensor_table(vnir, vnir_features, "VNIR")
    swir_ml = build_sensor_table(swir, swir_features, "SWIR")

    # Join at leaf level using metadata keys, not pixel coordinates.
    vnir_part = vnir[KEYS + vnir_features].copy()
    swir_part = swir[KEYS + swir_features].copy()

    vnir_part = vnir_part.rename(
        columns={c: f"VNIR_{c}" for c in vnir_features}
    )

    swir_part = swir_part.rename(
        columns={c: f"SWIR_{c}" for c in swir_features}
    )

    combined = vnir[METADATA_COLUMNS].merge(
        vnir_part,
        on=KEYS,
        validate="one_to_one"
    )

    combined = combined.merge(
        swir_part,
        on=KEYS,
        how="inner",
        validate="one_to_one"
    )

    combined = combined.sort_values(KEYS).reset_index(drop=True)

    combined_features = [
        c for c in combined.columns
        if c.startswith(("VNIR_", "SWIR_"))
    ]

    if combined[combined_features].isna().any().any():
        raise ValueError("Combined dataset contains missing features")

    output_path = OUTPUT_DIR / "multisensor_leaf_ml_dataset.csv"
    combined.to_csv(output_path, index=False)

    # Grouping audit: all leaves from the same pot must stay
    # in the same train/test partition.
    print("\nGROUPING AUDIT")
    print("-" * 55)

    print("Unique sample IDs:", combined["Sample_ID"].nunique())
    print("Unique pots:", combined["Pot"].nunique())
    print("Leaves:", len(combined))

    group_distribution = (
        combined.groupby("Pot")
        .agg(
            Leaf_Count=("Leaf", "count"),
            Infected_Leaves=("Disease_Label", "sum"),
            Treatment_Count=("Disease_Label", "nunique")
        )
        .reset_index()
    )

    conflicting_groups = group_distribution[
        group_distribution["Treatment_Count"] > 1
    ]

    print("Pots containing multiple disease labels:",
          len(conflicting_groups))

    if len(conflicting_groups):
        print(conflicting_groups.to_string(index=False))

    group_distribution.to_csv(
        OUTPUT_DIR / "pot_grouping_audit.csv",
        index=False
    )

    print("\nFINAL MULTISENSOR DATASET")
    print("-" * 55)
    print("Rows:", len(combined))
    print("VNIR features:", len(vnir_features))
    print("SWIR features:", len(swir_features))
    print("Combined spectral features:", len(combined_features))
    print("Infected:", int((combined["Disease_Label"] == 1).sum()))
    print("Non-infected:", int((combined["Disease_Label"] == 0).sum()))

    print("\nSaved:", output_path)
    print("\nAll original spectral files remain unchanged.")
    print("=" * 75)
    print("MULTISENSOR ML DATASET BUILD COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
