from pathlib import Path
import csv
import numpy as np
from spectral import open_image
from PIL import Image

# Dataset location
folder = Path(r"E:\Infotact_projects\Hyperspectral\dataset\raw\VNIR\13_G")

print("=" * 60)
print("TERRASPECTRA - VNIR SAMPLE INSPECTION")
print("=" * 60)

# 1. Open hyperspectral image using its HDR file
hdr_path = folder / "13_G.hdr"

image = open_image(str(hdr_path))

print("\n1. HYPERSPECTRAL IMAGE")
print("Shape:", image.shape)
print("Metadata:", image.metadata)

# 2. Memory-mapped access
cube = image.open_memmap()

print("\n2. CUBE INFORMATION")
print("Cube shape:", cube.shape)
print("Data type:", cube.dtype)
print("Memory mapped:", isinstance(cube, np.memmap))

# 3. Read only a small spatial region
small_region = np.asarray(cube[0:100, 0:100, :])

print("\n3. SAMPLE REGION")
print("Region shape:", small_region.shape)
print("Minimum:", small_region.min())
print("Maximum:", small_region.max())
print("Mean:", small_region.mean())
print("Contains NaN:", np.isnan(small_region).any())

# 4. Wavelength information
wavelengths = image.metadata.get("wavelength", [])

print("\n4. SPECTRAL INFORMATION")
print("Number of wavelengths:", len(wavelengths))

if wavelengths:
    print("First 10 wavelengths:", wavelengths[:10])
    print("First wavelength:", wavelengths[0])
    print("Last wavelength:", wavelengths[-1])

# 5. RGB reference image
jpg_path = folder / "13_G.jpg"

rgb = Image.open(jpg_path)

print("\n5. RGB REFERENCE")
print("Image dimensions:", rgb.size)
print("Image mode:", rgb.mode)

# 6. Inspect annotation CSV files
print("\n6. ANNOTATION FILES")

for csv_path in sorted(folder.glob("*.csv")):
    print("\nFile:", csv_path.name)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        for i, row in enumerate(reader):
            print(row)
            if i == 4:
                break

print("\nInspection completed successfully.")
