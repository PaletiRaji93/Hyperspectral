
from pathlib import Path

BASE_DIR = Path(r"E:\Infotact_projects\Hyperspectral")

samples = {
    "VNIR": BASE_DIR / "dataset" / "raw" / "VNIR" / "13_G" / "13_G.hdr",
    "SWIR": BASE_DIR / "dataset" / "raw" / "SWIR" / "13_G" / "13_G.hdr"
}

keywords = [
    "date",
    "time",
    "acquisition",
    "description",
    "sensor",
    "frame",
    "integration"
]

for sensor, header_path in samples.items():

    print("\n" + "=" * 80)
    print(sensor, "ACQUISITION HEADER")
    print("=" * 80)

    lines = header_path.read_text(
        encoding="latin1"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):

        if any(word in line.lower() for word in keywords):
            print(f"Line {line_number}: {line}")

# Print documentation manifest
manifest = BASE_DIR / "dataset" / "metadata" / "MANIFEST.TXT"

print("\n" + "=" * 80)
print("DATASET MANIFEST")
print("=" * 80)

print(manifest.read_text(encoding="cp1252"))
# ==================================================
# COMPLETE HEADER INSPECTION
# ==================================================

for sensor in ["VNIR", "SWIR"]:

    header_path = (
        BASE_DIR / "dataset" / "raw" /
        sensor / "13_G" / "13_G.hdr"
    )

    print("\n" + "=" * 80)
    print(sensor, "COMPLETE HEADER")
    print("=" * 80)

    with open(header_path, "r", encoding="latin1") as file:
        for line_number, line in enumerate(file, start=1):
            print(f"{line_number}: {line.rstrip()}")