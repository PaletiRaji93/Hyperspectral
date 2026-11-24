import os
import numpy as np
from PIL import Image

BASE = r"E:\Infotact_projects\Hyperspectral"
FOLDER = os.path.join(BASE, "dataset", "raw", "VNIR", "13_G")

# Load RGB image and align it with the hyperspectral cube
rgb = np.array(Image.open(os.path.join(FOLDER, "13_G.jpg")).convert("RGB"))
rgb = np.transpose(rgb, (1, 0, 2))

height, width = rgb.shape[:2]


def create_overlay(csv_name, candidate):
    csv_path = os.path.join(FOLDER, csv_name)
    points = np.loadtxt(csv_path, delimiter=",", dtype=int, ndmin=2)

    mask = np.zeros((height, width), dtype=bool)

    if candidate == "A":
        x = points[:, 0]
        y = points[:, 1]
    else:
        x = points[:, 1]
        y = points[:, 0]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    x = x[valid]
    y = y[valid]

    mask[y, x] = True

    # Highlight annotated pixels in red
    overlay = rgb.copy()
    overlay[mask] = [255, 0, 0]

    result = Image.blend(Image.fromarray(rgb), Image.fromarray(overlay), 0.65)

    output = os.path.join(
        BASE, f"{csv_name.replace('.csv', '')}_candidate_{candidate}.png"
    )

    result.save(output)

    print(f"Candidate {candidate}: {output}")
    print(f"Annotated pixels: {mask.sum()}")
    print(f"Valid coordinates: {len(x)}")


for leaf in ["13_G_leaf1.csv", "13_G_leaf2.csv"]:
    create_overlay(leaf, "A")
    create_overlay(leaf, "B")

print("\nComparison images generated successfully.")
