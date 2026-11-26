
import pandas as pd
import csv
from pathlib import Path

BASE_DIR = Path(r"E:\Infotact_projects\Hyperspectral")
META_DIR = BASE_DIR / "dataset" / "metadata" / "Data annotation"

# --------------------------------------------------
# 1. DISEASE DESCRIPTION
# --------------------------------------------------

print("\n" + "=" * 90)
print("DISEASE DESCRIPTION")
print("=" * 90)

description = pd.read_csv(
    META_DIR / "Data_description.csv",
    encoding="cp1252",
    sep=";",
    skiprows=2
)

description.columns = description.columns.str.strip()

print("Shape:", description.shape)
print("Columns:", description.columns.tolist())

# Remove completely empty rows
description = description.dropna(how="all")

print("\nTreatment counts:")
print(description["Treatment [5]"].value_counts(dropna=False))

print("\nConfidence-level counts:")
print(description["Confidence level [6]"].value_counts(dropna=False))

print("\nPot 13 records:")
pot13 = description[
    description["Pot [1]"].astype(str).str.strip().isin(["13", "13.0"])
]
print(pot13.to_string(index=False))


# --------------------------------------------------
# 2. VNIR AND SWIR ACQUISITION TABLES
# --------------------------------------------------

for sensor in ["VNIR", "SWIR"]:

    filename = f"{sensor}_data.csv"
    filepath = META_DIR / filename

    print("\n" + "=" * 90)
    print(sensor, "ACQUISITION METADATA")
    print("=" * 90)

    with open(filepath, "r", encoding="cp1252", newline="") as file:
        reader = csv.reader(file, delimiter=";", quotechar='"')
        rows = list(reader)

    date_header = rows[2]
    side_header = rows[3]

    print("Total rows:", len(rows))
    print("Total columns in date header:", len(date_header))
    print("Total columns in side header:", len(side_header))

    # Identify date and side for each acquisition column
    current_date = None
    acquisition_columns = []

    for index in range(3, len(side_header)):

        if index < len(date_header) and date_header[index].strip():
            current_date = date_header[index].strip()

        side = side_header[index].strip()

        acquisition_columns.append({
            "column_index": index,
            "dpi": current_date,
            "side": side
        })

    print("\nDATE AND SIDE COLUMN MAPPING:")

    for item in acquisition_columns:
        print(item)

    # Find pot 13
    print("\nPOT 13 ACQUISITION RECORD:")

    for row in rows[4:]:

        if not row or len(row) < 3:
            continue

        pot = row[0].strip()

        if pot == "13":
            print("Pot:", pot)
            print("Genotype:", row[1])
            print("Treatment:", row[2])

            for item in acquisition_columns:
                idx = item["column_index"]
                value = row[idx].strip() if idx < len(row) else ""

                print(
                    f"DPI={item['dpi']}, "
                    f"Side={item['side']}, "
                    f"Leaves={value if value else 'EMPTY'}"
                )

print("\nMetadata inspection completed.")
