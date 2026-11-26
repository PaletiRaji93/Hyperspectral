
import pandas as pd
import re
from pathlib import Path

# ============================================================
# TERRASPECTRA - MASTER SAMPLE METADATA BUILDER
# ============================================================

BASE_DIR = Path(r"E:\Infotact_projects\Hyperspectral")

META_DIR = BASE_DIR / "dataset" / "metadata" / "Data annotation"
RAW_DIR = BASE_DIR / "dataset" / "raw"

OUTPUT_DIR = BASE_DIR / "artifacts" / "dataset_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Provisional mapping - not yet independently confirmed
DATE_TO_DPI = {
    "2021-03-23": 11
}

# ============================================================
# 1. LOAD DISEASE DESCRIPTION
# ============================================================

print("\nLoading disease description...")

description = pd.read_csv(
    META_DIR / "Data_description.csv",
    encoding="cp1252",
    sep=";",
    skiprows=2
)

description = description.dropna(how="all")
description.columns = description.columns.str.strip()

description["Pot [1]"] = (
    pd.to_numeric(description["Pot [1]"], errors="coerce")
    .astype("Int64")
)

description["Side [2]"] = (
    description["Side [2]"].astype(str).str.strip()
)

description["Leaf [3]"] = (
    description["Leaf [3]"].astype(str).str.strip()
)

description["Genotype [4]"] = (
    description["Genotype [4]"].astype(str).str.strip()
)

description["Treatment [5]"] = (
    description["Treatment [5]"].astype(str).str.strip()
)

description["Confidence level [6]"] = (
    pd.to_numeric(
        description["Confidence level [6]"],
        errors="coerce"
    )
)

# ============================================================
# 2. INSPECT SENSOR FILES
# ============================================================

def inspect_sensor(sensor):

    sensor_dir = RAW_DIR / sensor
    available = {}

    if not sensor_dir.exists():
        print(f"WARNING: {sensor_dir} does not exist.")
        return available

    for sample_dir in sensor_dir.iterdir():

        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name
        hdr_file = sample_dir / f"{sample_id}.hdr"

        if not hdr_file.exists():
            continue

        acquisition_date = None

        with open(hdr_file, "r", encoding="latin1") as file:

            for line in file:

                match = re.search(
                    r"acquisition date\s*=\s*(\d{4}-\d{2}-\d{2})",
                    line,
                    re.IGNORECASE
                )

                if match:
                    acquisition_date = match.group(1)
                    break

        leaf_files = {}

        for leaf_file in sample_dir.glob("*_leaf*.csv"):

            match = re.search(
                r"leaf(\d+)",
                leaf_file.stem,
                re.IGNORECASE
            )

            if match:
                leaf_number = int(match.group(1))
                leaf_files[leaf_number] = True

        available[sample_id] = {
            "date": acquisition_date,
            "hdr_available": True,
            "leaves": leaf_files
        }

    return available


print("\nInspecting VNIR files...")
vnir_data = inspect_sensor("VNIR")
print("VNIR sample folders:", len(vnir_data))

print("\nInspecting SWIR files...")
swir_data = inspect_sensor("SWIR")
print("SWIR sample folders:", len(swir_data))

# ============================================================
# 3. BUILD MASTER RECORDS
# ============================================================

print("\nBuilding master metadata...")

all_sample_ids = sorted(set(vnir_data) | set(swir_data))

records = []

for sample_id in all_sample_ids:

    match = re.fullmatch(r"(\d+)_([GD])", sample_id)

    if not match:
        print("Skipping unexpected sample ID:", sample_id)
        continue

    pot_number = int(match.group(1))
    side = match.group(2)

    vnir = vnir_data.get(sample_id, {})
    swir = swir_data.get(sample_id, {})

    dates = [
        date for date in [vnir.get("date"), swir.get("date")]
        if date
    ]

    unique_dates = sorted(set(dates))

    if len(unique_dates) > 1:
        print(
            "WARNING: VNIR/SWIR date mismatch:",
            sample_id,
            unique_dates
        )

    acquisition_date = (
        unique_dates[0] if len(unique_dates) == 1 else None
    )

    dpi = DATE_TO_DPI.get(acquisition_date)

    dpi_status = (
        "Provisional" if dpi is not None else "Not mapped"
    )

    matching_records = description[
        (description["Pot [1]"] == pot_number)
        &
        (description["Side [2]"] == side)
    ]

    if matching_records.empty:
        print("WARNING: No disease metadata:", sample_id)
        continue

    for _, item in matching_records.iterrows():

        leaf_match = re.search(
            r"(\d+)",
            item["Leaf [3]"]
        )

        if not leaf_match:
            continue

        leaf_number = int(leaf_match.group(1))

        vnir_leaf_available = (
            leaf_number in vnir.get("leaves", {})
        )

        swir_leaf_available = (
            leaf_number in swir.get("leaves", {})
        )

        records.append({

            "Sample_ID": sample_id,
            "Pot": pot_number,
            "Side": side,
            "Leaf": leaf_number,

            "Acquisition_Date": acquisition_date,
            "DPI": dpi,
            "DPI_Mapping_Status": dpi_status,

            "Genotype": item["Genotype [4]"],
            "Treatment": item["Treatment [5]"],
            "Confidence": item["Confidence level [6]"],

            "VNIR_Available": sample_id in vnir_data,
            "SWIR_Available": sample_id in swir_data,

            "VNIR_Leaf_Annotation": vnir_leaf_available,
            "SWIR_Leaf_Annotation": swir_leaf_available
        })


# ============================================================
# 4. CREATE DATAFRAME
# ============================================================

master_df = pd.DataFrame(records)

if master_df.empty:
    raise ValueError(
        "No metadata records generated. Check dataset paths and CSV files."
    )

# ============================================================
# 5. ANNOTATION ELIGIBILITY
# ============================================================

def annotation_status(row):

    vnir = row["VNIR_Leaf_Annotation"]
    swir = row["SWIR_Leaf_Annotation"]

    if vnir and swir:
        return "Available_Both_Sensors"

    if not vnir and not swir:
        return "Missing_Both_Sensors"

    if not vnir:
        return "Missing_VNIR_Annotation"

    return "Missing_SWIR_Annotation"


master_df["Annotation_Status"] = master_df.apply(
    annotation_status,
    axis=1
)

master_df["Eligible_For_Leaf_Analysis"] = (
    master_df["VNIR_Leaf_Annotation"]
    & master_df["SWIR_Leaf_Annotation"]
)

# ============================================================
# 6. SAVE MASTER CSV
# ============================================================

output_file = OUTPUT_DIR / "master_sample_metadata.csv"

master_df.to_csv(output_file, index=False)

# ============================================================
# 7. VALIDATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("TERRASPECTRA MASTER METADATA REPORT")
print("=" * 80)

print("\nTotal metadata records:", len(master_df))

print(
    "Unique samples:",
    master_df["Sample_ID"].nunique()
)

print(
    "\nEligible leaf records:",
    int(master_df["Eligible_For_Leaf_Analysis"].sum())
)

print("\nAnnotation status:")
print(master_df["Annotation_Status"].value_counts())

print("\nTreatment distribution:")
print(master_df["Treatment"].value_counts(dropna=False))

print("\nGenotype distribution:")
print(master_df["Genotype"].value_counts(dropna=False))

print("\nConfidence distribution:")
print(master_df["Confidence"].value_counts(dropna=False))

print("\nAcquisition dates:")
print(master_df["Acquisition_Date"].value_counts(dropna=False))

print("\nDPI distribution:")
print(master_df["DPI"].value_counts(dropna=False))

print("\nDPI mapping status:")
print(master_df["DPI_Mapping_Status"].value_counts())

print("\nSensor availability:")
print(
    master_df[
        ["VNIR_Available", "SWIR_Available"]
    ].drop_duplicates().value_counts()
)

print("\nMissing VNIR annotations:")
print((~master_df["VNIR_Leaf_Annotation"]).sum())

print("\nMissing SWIR annotations:")
print((~master_df["SWIR_Leaf_Annotation"]).sum())

print("\nEligible treatment distribution:")
eligible_df = master_df[
    master_df["Eligible_For_Leaf_Analysis"]
]

print(eligible_df["Treatment"].value_counts())

print("\nFirst 15 records:")
print(master_df.head(15).to_string(index=False))

print("\nSaved to:")
print(output_file)

print("\nMetadata generation completed successfully.")
