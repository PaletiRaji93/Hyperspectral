
from pathlib import Path
from collections import Counter

ROOT = Path("dataset/raw/VNIR")

print("=" * 75)
print("TERRASPECTRA - COMPLETE VNIR DATASET INSPECTION")
print("=" * 75)

if not ROOT.exists():
    raise FileNotFoundError(ROOT)

# -----------------------------------------------------
# 1. DIRECTORY INFORMATION
# -----------------------------------------------------

print("\n1. VNIR DIRECTORY")

print("Root:", ROOT.resolve())

folders = sorted(p for p in ROOT.iterdir() if p.is_dir())

print("Number of sample directories:", len(folders))

# -----------------------------------------------------
# 2. FILE EXTENSION SUMMARY
# -----------------------------------------------------

print("\n2. FILE EXTENSION SUMMARY")

extension_counts = Counter()

for path in ROOT.rglob("*"):
    if path.is_file():
        extension_counts[path.suffix.lower()] += 1

for extension, count in sorted(extension_counts.items()):
    print(f"{extension or '[no extension]':20} {count}")

# -----------------------------------------------------
# 3. SAMPLE DIRECTORY CONTENTS
# -----------------------------------------------------

print("\n3. SAMPLE DIRECTORY CONTENTS")

for folder in folders:

    print(f"\n[{folder.name}]")

    for file in sorted(folder.iterdir()):

        if file.is_file():

            size_mb = file.stat().st_size / (1024 * 1024)

            print(f"  {file.name:40} {size_mb:10.2f} MB")

# -----------------------------------------------------
# 4. SEARCH REFERENCE-RELATED FILES
# -----------------------------------------------------

print("\n4. REFERENCE / BACKGROUND / CALIBRATION FILES")

keywords = [
    "dark",
    "white",
    "background",
    "reference",
    "calib",
    "reflectance",
    "radiance",
    "standard",
    "panel"
]

matches = []

for path in ROOT.rglob("*"):

    if not path.is_file():
        continue

    if any(word in path.name.lower() for word in keywords):
        matches.append(path)

if matches:

    for path in matches:
        print(path)

else:
    print("No matching filenames found.")

# -----------------------------------------------------
# 5. INSPECT SAMPLE 13_G HEADER
# -----------------------------------------------------

print("\n5. COMPLETE 13_G HEADER")

header = ROOT / "13_G" / "13_G.hdr"

if header.exists():

    print(header.read_text(encoding="utf-8", errors="replace"))

else:

    print("13_G.hdr not found.")

# -----------------------------------------------------
# 6. SAMPLE FILE COUNTS
# -----------------------------------------------------

print("\n6. SAMPLE COMPLETENESS")

for folder in folders:

    files = list(folder.iterdir())

    hdr_count = sum(
        p.suffix.lower() == ".hdr" for p in files
    )

    hyspex_count = sum(
        p.suffix.lower() == ".hyspex" for p in files
    )

    csv_count = sum(
        p.suffix.lower() == ".csv" for p in files
    )

    print(
        f"{folder.name:20} "
        f"HDR={hdr_count} "
        f"HYSPEX={hyspex_count} "
        f"CSV={csv_count}"
    )

print("\n" + "=" * 75)
print("VNIR DATASET INSPECTION COMPLETED")
print("=" * 75)
