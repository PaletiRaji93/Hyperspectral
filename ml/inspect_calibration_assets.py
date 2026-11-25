
from pathlib import Path
from spectral import open_image

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT = Path(".")
SAMPLE_FOLDER = PROJECT / "dataset/raw/VNIR/13_G"
HDR_PATH = SAMPLE_FOLDER / "13_G.hdr"

KEYWORDS = [
    "calib",
    "calibration",
    "dark",
    "white",
    "reference",
    "reflectance",
    "radiometric",
    "correction",
    "background",
    "response"
]

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "hyperenv",
    "artifacts"
}

print("=" * 70)
print("TERRASPECTRA - CALIBRATION ASSET INVESTIGATION")
print("=" * 70)

# =====================================================
# 2. INSPECT SAMPLE DIRECTORY
# =====================================================

print("\n1. FILES IN SAMPLE DIRECTORY")

for path in sorted(SAMPLE_FOLDER.iterdir()):

    if path.is_file():
        print(
            f"{path.name:45} "
            f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        )
    else:
        print(f"{path.name}/")

# =====================================================
# 3. INSPECT CALIBRATION-RELATED FILES
# =====================================================

print("\n2. CALIBRATION-RELATED FILES IN PROJECT")

found = []

for path in PROJECT.rglob("*"):

    if not path.is_file():
        continue

    if any(part in SKIP_DIRS for part in path.parts):
        continue

    if any(word in path.name.lower() for word in KEYWORDS):

        found.append(path)

for path in found:

    print(
        path,
        f"({path.stat().st_size / (1024 * 1024):.2f} MB)"
    )

if not found:
    print("No calibration-related filenames found.")

# =====================================================
# 4. INSPECT ENVI HEADER METADATA
# =====================================================

print("\n3. ENVI HEADER METADATA")

image = open_image(str(HDR_PATH))
metadata = image.metadata

important_fields = [
    "description",
    "data type",
    "data ignore value",
    "wavelength units",
    "wavelength",
    "default bands",
    "interleave",
    "acquisition date",
    "acquisition start time"
]

for field in important_fields:

    if field not in metadata:
        continue

    value = metadata[field]

    if field == "wavelength":
        print(
            f"{field}: {len(value)} wavelength values"
        )
    else:
        print(f"\n{field}: {value}")

# =====================================================
# 5. INSPECT DESCRIPTION FOR CALIBRATION INFORMATION
# =====================================================

print("\n4. CALIBRATION INFORMATION IN DESCRIPTION")

description = str(metadata.get("description", ""))

for line in description.splitlines():

    if any(word in line.lower() for word in KEYWORDS):
        print(line)

# =====================================================
# 6. FINAL SUMMARY
# =====================================================

print("\n5. INVESTIGATION SUMMARY")

print("Calibration-related files found:", len(found))

print("\nInvestigation completed.")
print("=" * 70)
