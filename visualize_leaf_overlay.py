
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path("dataset/raw/VNIR/13_G")
OUTPUT = Path("outputs/leaf_masks")

OUTPUT.mkdir(parents=True, exist_ok=True)

hdr_shape = (1370, 1024)

# --------------------------------------------------
# 2. LOAD ORIGINAL RGB IMAGE
# --------------------------------------------------

rgb_path = ROOT / "13_G.jpg"

rgb = np.asarray(
    Image.open(rgb_path).convert("RGB")
)

print("=" * 60)
print("TERRASPECTRA - RGB LEAF ANNOTATION OVERLAY")
print("=" * 60)

print("\nRGB image shape:", rgb.shape)
print("Expected hyperspectral shape:", hdr_shape)

# --------------------------------------------------
# 3. LOAD CSV COORDINATES
# --------------------------------------------------

leaf_files = {
    "leaf1": ROOT / "13_G_leaf1.csv",
    "leaf2": ROOT / "13_G_leaf2.csv"
}

masks = {}

for name, path in leaf_files.items():

    df = pd.read_csv(path, header=None)

    rows = df.iloc[:, 0].to_numpy(dtype=int)
    cols = df.iloc[:, 1].to_numpy(dtype=int)

    mask = np.zeros(hdr_shape, dtype=np.uint8)

    mask[rows, cols] = 1

    masks[name] = mask

    print(f"\n{name}: {mask.sum()} annotated pixels")

# --------------------------------------------------
# 4. DETERMINE RGB ORIENTATION
# --------------------------------------------------

if rgb.shape[:2] == hdr_shape:

    print("\nRGB orientation: Original")

    aligned_masks = masks

elif rgb.shape[:2] == hdr_shape[::-1]:

    print("\nRGB orientation: Transposed")

    aligned_masks = {
        name: mask.T
        for name, mask in masks.items()
    }

else:

    raise ValueError(
        "RGB dimensions do not match the hyperspectral cube "
        "or its transpose. Check image dimensions."
    )

# --------------------------------------------------
# 5. CREATE COMBINED MASK
# --------------------------------------------------

combined = np.zeros(
    rgb.shape[:2],
    dtype=np.uint8
)

combined[aligned_masks["leaf1"] == 1] = 1
combined[aligned_masks["leaf2"] == 1] = 2

print("\nCombined annotation statistics:")
print("Leaf 1:", np.sum(combined == 1))
print("Leaf 2:", np.sum(combined == 2))
print("Background:", np.sum(combined == 0))

# --------------------------------------------------
# 6. CREATE COLORED OVERLAY
# --------------------------------------------------

overlay = rgb.astype(np.float32).copy()

colors = {
    1: np.array([0, 100, 255]),
    2: np.array([255, 120, 0])
}

alpha = 0.65

for label, color in colors.items():

    selected = combined == label

    overlay[selected] = (
        (1 - alpha) * overlay[selected]
        + alpha * color
    )

overlay = np.clip(overlay, 0, 255).astype(np.uint8)

# --------------------------------------------------
# 7. VISUALIZE
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

axes[0].imshow(rgb)
axes[0].set_title("Original RGB")

axes[1].imshow(combined, cmap="viridis")
axes[1].set_title("Combined Leaf Mask")

axes[2].imshow(overlay)
axes[2].set_title("RGB + Leaf Annotation Overlay")

for ax in axes:
    ax.axis("off")

from matplotlib.patches import Patch

legend = [
    Patch(facecolor="blue", label="Leaf 1"),
    Patch(facecolor="orange", label="Leaf 2")
]

axes[2].legend(handles=legend, loc="lower right")

plt.tight_layout()

plt.savefig(
    OUTPUT / "13_G_rgb_annotation_overlay.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nOverlay saved successfully!")
print(OUTPUT / "13_G_rgb_annotation_overlay.png")

