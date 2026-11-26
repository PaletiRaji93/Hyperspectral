
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "13_G"

VNIR_DIR = ROOT / "dataset/raw/VNIR" / SAMPLE
SWIR_DIR = ROOT / "dataset/raw/SWIR" / SAMPLE

OUT_DIR = ROOT / "outputs/enhanced_references"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_enhanced_reference(folder, sensor):
    path = folder / f"{SAMPLE}.jpg"

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(path)

    # Correct reference orientation
    image = image.T.copy()

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    # Additional percentile normalization
    low, high = np.percentile(enhanced, [1, 99])

    enhanced = np.clip(
        (enhanced.astype(np.float32) - low)
        * 255.0 / max(high - low, 1),
        0,
        255
    ).astype(np.uint8)

    print(f"{sensor} image shape:", enhanced.shape)

    return enhanced


def create_leaf_mask(folder, leaf, shape):
    csv_path = folder / f"{SAMPLE}_{leaf}.csv"

    coords = pd.read_csv(
        csv_path,
        header=None
    ).iloc[:, :2].to_numpy(dtype=int)

    mask = np.zeros(shape, dtype=np.uint8)

    rows = coords[:, 0]
    cols = coords[:, 1]

    valid = (
        (rows >= 0) & (rows < shape[0]) &
        (cols >= 0) & (cols < shape[1])
    )

    mask[rows[valid], cols[valid]] = 255

    return mask


def create_overlay(image, folder):
    # Convert grayscale image to color
    overlay = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    # Leaf 1 = red
    # Leaf 2 = blue
    colors = {
        "leaf1": (0, 0, 255),
        "leaf2": (255, 0, 0)
    }

    for leaf, color in colors.items():
        mask = create_leaf_mask(
            folder,
            leaf,
            image.shape
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            2
        )

    return cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )


# Load enhanced images
vnir = load_enhanced_reference(VNIR_DIR, "VNIR")
swir = load_enhanced_reference(SWIR_DIR, "SWIR")

# Draw annotation boundaries
vnir_overlay = create_overlay(vnir, VNIR_DIR)
swir_overlay = create_overlay(swir, SWIR_DIR)

# Display and save
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].imshow(vnir, cmap="gray")
axes[0, 0].set_title("Enhanced VNIR reference")

axes[0, 1].imshow(vnir_overlay)
axes[0, 1].set_title("VNIR with leaf annotations")

axes[1, 0].imshow(swir, cmap="gray")
axes[1, 0].set_title("Enhanced SWIR reference")

axes[1, 1].imshow(swir_overlay)
axes[1, 1].set_title("SWIR with leaf annotations")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

output = OUT_DIR / f"{SAMPLE}_enhanced_leaf_annotations.png"

plt.savefig(
    output,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print("\nEnhanced visualization saved:")
print(output)
print("\nLeaf colors:")
print("Red  = Leaf 1")
print("Blue = Leaf 2")
