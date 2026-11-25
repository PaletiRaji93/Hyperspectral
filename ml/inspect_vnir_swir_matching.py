
from pathlib import Path
import re
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VNIR_ROOT = PROJECT_ROOT / "dataset" / "raw" / "VNIR"
SWIR_ROOT = PROJECT_ROOT / "dataset" / "raw" / "SWIR"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dataset_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_metadata(hdr_path):
    text = hdr_path.read_text(errors="ignore")

    def get_value(pattern):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else "Not found"

    return {
        "sample_id": hdr_path.stem,
        "hdr_path": str(hdr_path),
        "lines": get_value(r"^\s*lines\s*=\s*(\d+)"),
        "samples": get_value(r"^\s*samples\s*=\s*(\d+)"),
        "bands": get_value(r"^\s*bands\s*=\s*(\d+)"),
        "interleave": get_value(r"^\s*interleave\s*=\s*(\S+)"),
        "acquisition_date": get_value(
            r"^\s*acquisition date\s*=\s*([^\r\n]+)"
        ),
        "acquisition_start": get_value(
            r"^[ \t]*acquisition start[ \t]*=[ \t]*([^\r\n]+)"
        ),
        "wavelength_units": get_value(
            r"^\s*wavelength units\s*=\s*([^\r\n]+)"
        ),
    }


def inspect_dataset(name, root):
    print("\n" + "=" * 65)
    print(f"{name} DATASET INSPECTION")
    print("=" * 65)

    if not root.exists():
        print(f"ERROR: Folder not found: {root}")
        return []

    hdr_files = list(root.rglob("*.hdr"))
    hyspex_files = list(root.rglob("*.hyspex"))
    csv_files = list(root.rglob("*.csv"))
    jpg_files = list(root.rglob("*.jpg"))

    print("Root:", root)
    print("HDR files:", len(hdr_files))
    print("HYSPEX files:", len(hyspex_files))
    print("CSV files:", len(csv_files))
    print("JPG files:", len(jpg_files))

    records = []

    for hdr in sorted(hdr_files):
        metadata = extract_metadata(hdr)
        records.append(metadata)

    print("\nFIRST 15 SAMPLE HEADERS")

    for item in records[:15]:
        print("-" * 45)
        print("Sample:", item["sample_id"])
        print("Dimensions:", item["lines"], item["samples"])
        print("Bands:", item["bands"])
        print("Date:", item["acquisition_date"])
        print("Start:", item["acquisition_start"])

    return records


vnir_records = inspect_dataset("VNIR", VNIR_ROOT)
swir_records = inspect_dataset("SWIR", SWIR_ROOT)


# Match samples using their header filenames
vnir_ids = {x["sample_id"] for x in vnir_records}
swir_ids = {x["sample_id"] for x in swir_records}

common_ids = sorted(vnir_ids & swir_ids)

print("\n" + "=" * 65)
print("VNIR - SWIR SAMPLE MATCHING")
print("=" * 65)

print("VNIR unique sample IDs:", len(vnir_ids))
print("SWIR unique sample IDs:", len(swir_ids))
print("Common sample IDs:", len(common_ids))

print("\nFirst 20 common samples:")
for sample in common_ids[:20]:
    print(sample)

print("\nVNIR-only sample count:", len(vnir_ids - swir_ids))
print("SWIR-only sample count:", len(swir_ids - vnir_ids))


# Save metadata reports
for name, records in [
    ("VNIR", vnir_records),
    ("SWIR", swir_records)
]:
    output_file = OUTPUT_DIR / f"{name.lower()}_header_report.csv"

    if records:
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        print(f"\nSaved: {output_file}")


# Save matching IDs
match_file = OUTPUT_DIR / "common_vnir_swir_samples.csv"

with match_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id"])
    for sample in common_ids:
        writer.writerow([sample])

print("Saved:", match_file)
print("\nINSPECTION COMPLETED")
