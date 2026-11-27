
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import spectral


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    ROOT / "artifacts/dataset_inspection/master_sample_metadata.csv"
)

OUTPUT_DIR = ROOT / "outputs/leaf_level_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = {
    "VNIR": {
        "folder": ROOT / "dataset/raw/VNIR",
        "ignore_value": 4095
    },
    "SWIR": {
        "folder": ROOT / "dataset/raw/SWIR",
        "ignore_value": 16383
    }
}


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def calculate_leaf_features(spectra, wavelengths, sensor, metadata):
    """
    Calculate statistical spectral features for one leaf.
    The ENVI ignore value is excluded from spectral statistics.
    SWIR 65535 values are retained and separately counted because
    their physical meaning has not yet been confirmed.
    """

    spectra = spectra.astype(np.float32, copy=True)

    ignore_value = SENSORS[sensor]["ignore_value"]

    ignore_count = int(np.sum(spectra == ignore_value))
    zero_count = int(np.sum(spectra == 0))
    maximum_count = int(np.sum(spectra == 65535))

    spectra[spectra == ignore_value] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        mean_values = np.nanmean(spectra, axis=0)
        std_values = np.nanstd(spectra, axis=0)
        median_values = np.nanmedian(spectra, axis=0)

        q25_values = np.nanpercentile(spectra, 25, axis=0)
        q75_values = np.nanpercentile(spectra, 75, axis=0)

    feature = {
        "Sample_ID": str(metadata["Sample_ID"]),
        "Pot": metadata["Pot"],
        "Side": metadata["Side"],
        "Leaf": int(metadata["Leaf"]),
        "Acquisition_Date": metadata["Acquisition_Date"],
        "DPI": metadata["DPI"],
        "DPI_Mapping_Status": metadata["DPI_Mapping_Status"],
        "Genotype": metadata["Genotype"],
        "Treatment": metadata["Treatment"],
        "Disease_Label": (
            1 if metadata["Treatment"] == "Infected" else 0
        ),
        "Confidence": metadata["Confidence"],
        "Sensor": sensor,
        "Pixel_Count": len(spectra),
        "Ignore_Value_Count": ignore_count,
        "Zero_Count": zero_count,
        "Maximum_Value_Count": maximum_count
    }

    for i, wavelength in enumerate(wavelengths):
        band = f"{wavelength:.2f}nm"

        feature[f"mean_{band}"] = mean_values[i]
        feature[f"std_{band}"] = std_values[i]
        feature[f"median_{band}"] = median_values[i]
        feature[f"q25_{band}"] = q25_values[i]
        feature[f"q75_{band}"] = q75_values[i]

    return feature


def process_sensor(sensor, config, metadata):
    print("\n" + "=" * 75)
    print(f"PROCESSING COMPLETE {sensor} DATASET")
    print("=" * 75)

    sensor_folder = config["folder"]
    annotation_column = f"{sensor}_Leaf_Annotation"

    sensor_metadata = metadata[
        metadata["Eligible_For_Leaf_Analysis"].apply(to_bool)
        & metadata[annotation_column].apply(to_bool)
    ].copy()

    print("Eligible metadata records:", len(sensor_metadata))
    print("Unique sample folders:", sensor_metadata["Sample_ID"].nunique())

    feature_records = []
    issues = []

    sample_groups = sensor_metadata.groupby("Sample_ID")

    for sample_index, (sample_id, group) in enumerate(
        sample_groups, start=1
    ):
        sample_folder = sensor_folder / str(sample_id)
        header_path = sample_folder / f"{sample_id}.hdr"

        if not header_path.exists():
            issues.append({
                "Sample_ID": sample_id,
                "Issue": "Missing ENVI header"
            })
            continue

        try:
            image = spectral.open_image(str(header_path))
            cube = image.open_memmap()

            wavelengths = np.asarray(
                image.metadata["wavelength"],
                dtype=np.float32
            )

            if cube.ndim != 3:
                raise ValueError(f"Unexpected cube shape: {cube.shape}")

            if cube.shape[2] != len(wavelengths):
                raise ValueError("Wavelength count does not match cube")

            for _, row in group.iterrows():
                leaf_number = int(row["Leaf"])

                annotation_path = (
                    sample_folder /
                    f"{sample_id}_leaf{leaf_number}.csv"
                )

                if not annotation_path.exists():
                    issues.append({
                        "Sample_ID": sample_id,
                        "Leaf": leaf_number,
                        "Issue": "Missing leaf annotation"
                    })
                    continue

                coords = pd.read_csv(
                    annotation_path,
                    header=None
                ).iloc[:, :2].to_numpy(dtype=np.int64)

                if coords.ndim != 2 or coords.shape[1] != 2:
                    raise ValueError("Invalid annotation coordinates")

                rows = coords[:, 0]
                cols = coords[:, 1]

                if (
                    np.any(rows < 0)
                    or np.any(rows >= cube.shape[0])
                    or np.any(cols < 0)
                    or np.any(cols >= cube.shape[1])
                ):
                    raise ValueError(
                        f"Leaf {leaf_number}: coordinates outside cube"
                    )

                spectra = np.asarray(cube[rows, cols, :])

                feature = calculate_leaf_features(
                    spectra,
                    wavelengths,
                    sensor,
                    row
                )

                feature_records.append(feature)

        except Exception as error:
            issues.append({
                "Sample_ID": sample_id,
                "Issue": str(error)
            })

        if sample_index % 10 == 0:
            print(
                f"Processed {sample_index}/"
                f"{len(sample_groups)} sample folders"
            )

    if not feature_records:
        raise RuntimeError(f"No {sensor} features were extracted")

    result = pd.DataFrame.from_records(feature_records)

    output_path = (
        OUTPUT_DIR / f"{sensor.lower()}_all_leaf_features.csv"
    )

    result.to_csv(output_path, index=False)

    issue_path = (
        OUTPUT_DIR / f"{sensor.lower()}_extraction_issues.csv"
    )

    pd.DataFrame(issues).to_csv(issue_path, index=False)

    print("\n" + "-" * 60)
    print(f"{sensor} EXTRACTION SUMMARY")
    print("-" * 60)

    print("Extracted leaf records:", len(result))
    print("Unique sample IDs:", result["Sample_ID"].nunique())
    print("Infected leaves:", (result["Disease_Label"] == 1).sum())
    print("Non-infected leaves:", (result["Disease_Label"] == 0).sum())

    print("\nTreatment distribution:")
    print(result["Treatment"].value_counts())

    print("\nGenotype distribution:")
    print(result["Genotype"].value_counts())

    print("\nExtraction issues:", len(issues))
    print("Feature dataset:", output_path)
    print("Issue report:", issue_path)

    return result


def main():
    print("=" * 75)
    print("TERRASPECTRA - COMPLETE LEAF FEATURE EXTRACTION")
    print("=" * 75)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = [
        "Sample_ID",
        "Leaf",
        "Treatment",
        "Genotype",
        "Confidence",
        "Eligible_For_Leaf_Analysis",
        "VNIR_Leaf_Annotation",
        "SWIR_Leaf_Annotation"
    ]

    missing = [
        column for column in required_columns
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    results = {}

    for sensor, config in SENSORS.items():
        if not config["folder"].exists():
            raise FileNotFoundError(config["folder"])

        results[sensor] = process_sensor(
            sensor,
            config,
            metadata
        )

    print("\n" + "=" * 75)
    print("COMPLETE DATASET EXTRACTION SUMMARY")
    print("=" * 75)

    for sensor, result in results.items():
        print(
            f"{sensor}: {len(result)} leaf records, "
            f"{result['Sample_ID'].nunique()} sample folders"
        )

    print("\nOriginal hyperspectral data was not modified.")
    print("COMPLETE LEAF FEATURE EXTRACTION FINISHED")


if __name__ == "__main__":
    main()
