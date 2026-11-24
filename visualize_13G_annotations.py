from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(r"E:\Infotact_projects\Hyperspectral")
folder = ROOT / "dataset" / "raw" / "VNIR" / "13_G"

# Load RGB image
rgb = np.asarray(Image.open(folder / "13_G.jpg").convert("RGB"))

# Transpose RGB to match the hyperspectral cube's spatial shape
rgb_transposed = np.transpose(rgb, (1, 0, 2))

print("Original RGB shape:", rgb.shape)
print("Transposed RGB shape:", rgb_transposed.shape)
print("Expected cube spatial shape:", (1370, 1024))

# Load leaf coordinates
annotations = {}

for name in ["13_G_leaf1.csv", "13_G_leaf2.csv"]:
    path = folder / name
    points = np.loadtxt(path, delimiter=",", dtype=int, ndmin=2)
    annotations[name] = points

    print("\n", name)
    print("Coordinate records:", len(points))
    print("First 5 coordinates:\n", points[:5])
    print("Column 1 range:", points[:, 0].min(), points[:, 0].max())
    print("Column 2 range:", points[:, 1].min(), points[:, 1].max())

    # Test both possible coordinate interpretations
    first_is_x = (
        (points[:, 0] >= 0).all()
        and (points[:, 0] < 1024).all()
        and (points[:, 1] >= 0).all()
        and (points[:, 1] < 1370).all()
    )

    second_is_x = (
        (points[:, 1] >= 0).all()
        and (points[:, 1] < 1024).all()
        and (points[:, 0] >= 0).all()
        and (points[:, 0] < 1370).all()
    )

    print("Column 1 = x, column 2 = y valid:", first_is_x)
    print("Column 2 = x, column 1 = y valid:", second_is_x)


# Visual comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("Original RGB")

axes[0, 1].imshow(rgb_transposed)
axes[0, 1].set_title("Transposed RGB")

# Candidate A: first column is x, second is y
axes[1, 0].imshow(rgb_transposed)
for name, points in annotations.items():
    axes[1, 0].scatter(points[:, 0], points[:, 1], s=0.3, label=name)
axes[1, 0].set_title("Candidate A: (x, y)")
axes[1, 0].legend(markerscale=10)

# Candidate B: second column is x, first is y
axes[1, 1].imshow(rgb_transposed)
for name, points in annotations.items():
    axes[1, 1].scatter(points[:, 1], points[:, 0], s=0.3, label=name)
axes[1, 1].set_title("Candidate B: (y, x)")
axes[1, 1].legend(markerscale=10)

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

output = ROOT / "13_G_annotation_orientation_check.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved visualization:", output)
