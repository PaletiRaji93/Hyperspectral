
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"E:\Infotact_projects\Hyperspectral")

MASTER_FILE = (
    BASE_DIR / "artifacts" / "dataset_inspection"
    / "master_sample_metadata.csv"
)

RAW_DIR = BASE_DIR / "dataset" / "raw"

df = pd.read_csv(MASTER_FILE)

missing = df[
    (~df["VNIR_Leaf_Annotation"])
    | (~df["SWIR_Leaf_Annotation"])
]

print("\n" + "=" * 80)
print("MISSING LEAF ANNOTATION INVESTIGATION")
print("=" * 80)

print("\nMissing records:")
print(
    missing[
        ["Sample_ID", "Leaf", "Treatment", "Genotype",
         "VNIR_Leaf_Annotation", "SWIR_Leaf_Annotation"]
    ].to_string(index=False)
)

for _, record in missing.iterrows():

    sample_id = record["Sample_ID"]
    leaf = int(record["Leaf"])

    print("\n" + "-" * 70)
    print(f"SAMPLE: {sample_id} | LEAF: {leaf}")

    for sensor in ["VNIR", "SWIR"]:

        folder = RAW_DIR / sensor / sample_id

        print(f"\n{sensor} folder: {folder}")

        if not folder.exists():
            print("Folder does not exist.")
            continue

        print("Available files:")

        for file in sorted(folder.iterdir()):
            print(" ", file.name)

        expected = folder / f"{sample_id}_leaf{leaf}.csv"

        print(
            "Expected annotation:",
            "FOUND" if expected.exists() else "NOT FOUND"
        )

print("\nInvestigation completed.")
