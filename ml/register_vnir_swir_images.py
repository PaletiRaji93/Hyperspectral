
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "13_G"

VNIR_DIR = ROOT / "dataset" / "raw" / "VNIR" / SAMPLE
SWIR_DIR = ROOT / "dataset" / "raw" / "SWIR" / SAMPLE

OUT_DIR = ROOT / "outputs" / "registration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load reference images and correct orientation
# --------------------------------------------------

vnir = cv2.imread(str(VNIR_DIR / f"{SAMPLE}.jpg"),
                  cv2.IMREAD_GRAYSCALE)

swir = cv2.imread(str(SWIR_DIR / f"{SAMPLE}.jpg"),
                  cv2.IMREAD_GRAYSCALE)

if vnir is None or swir is None:
    raise FileNotFoundError("Could not load VNIR or SWIR reference JPG.")

# Both reference images were verified to require transpose
vnir = vnir.T.copy()
swir = swir.T.copy()

print("VNIR image shape:", vnir.shape)
print("SWIR image shape:", swir.shape)

# --------------------------------------------------
# 2. Detect and match image features
# --------------------------------------------------

if hasattr(cv2, "SIFT_create"):
    detector = cv2.SIFT_create(nfeatures=5000)
    norm_type = cv2.NORM_L2
    ratio = 0.75
    print("\nUsing SIFT feature detector.")
else:
    detector = cv2.ORB_create(nfeatures=5000)
    norm_type = cv2.NORM_HAMMING
    ratio = 0.80
    print("\nUsing ORB feature detector.")

kp1, des1 = detector.detectAndCompute(vnir, None)
kp2, des2 = detector.detectAndCompute(swir, None)

if des1 is None or des2 is None:
    raise RuntimeError("Not enough image features detected.")

matcher = cv2.BFMatcher(norm_type)

matches = matcher.knnMatch(des1, des2, k=2)

good_matches = [
    m for m, n in matches
    if m.distance < ratio * n.distance
]

print("VNIR keypoints:", len(kp1))
print("SWIR keypoints:", len(kp2))
print("Good feature matches:", len(good_matches))

if len(good_matches) < 10:
    raise RuntimeError(
        "Insufficient feature matches. Registration cannot be trusted."
    )

# --------------------------------------------------
# 3. Estimate transformation
# --------------------------------------------------

src_pts = np.float32(
    [kp1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [kp2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

H, inlier_mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

if H is None:
    raise RuntimeError("Could not estimate homography.")

inliers = int(inlier_mask.sum())
inlier_ratio = inliers / len(good_matches)

print("\nREGISTRATION RESULTS")
print("Inliers:", inliers)
print("Inlier ratio:", round(inlier_ratio, 4))
print("\nTransformation matrix:")
print(H)

# --------------------------------------------------
# 4. Transform VNIR image into SWIR coordinates
# --------------------------------------------------

height, width = swir.shape

warped_vnir = cv2.warpPerspective(
    vnir,
    H,
    (width, height)
)

# --------------------------------------------------
# 5. Load leaf annotations
# --------------------------------------------------

def load_mask(folder, leaf, shape):
    coords = pd.read_csv(
        folder / f"{SAMPLE}_{leaf}.csv",
        header=None
    ).iloc[:, :2].to_numpy(dtype=int)

    mask = np.zeros(shape, dtype=np.uint8)

    rows = coords[:, 0]
    cols = coords[:, 1]

    valid = (
        (rows >= 0) & (rows < shape[0]) &
        (cols >= 0) & (cols < shape[1])
    )

    mask[rows[valid], cols[valid]] = 1

    return mask

for leaf in ["leaf1", "leaf2"]:

    vnir_mask = load_mask(
        VNIR_DIR, leaf, vnir.shape
    )

    swir_mask = load_mask(
        SWIR_DIR, leaf, swir.shape
    )

    warped_mask = cv2.warpPerspective(
        vnir_mask,
        H,
        (width, height),
        flags=cv2.INTER_NEAREST
    )

    intersection = np.logical_and(
        warped_mask > 0,
        swir_mask > 0
    ).sum()

    union = np.logical_or(
        warped_mask > 0,
        swir_mask > 0
    ).sum()

    iou = intersection / union if union else 0

    print(f"\n{leaf.upper()} MASK COMPARISON")
    print("VNIR mask pixels:", int(vnir_mask.sum()))
    print("SWIR mask pixels:", int(swir_mask.sum()))
    print("Intersection:", int(intersection))
    print("Union:", int(union))
    print("IoU:", round(iou, 4))

# --------------------------------------------------
# 6. Save visual diagnostic
# --------------------------------------------------

overlay = cv2.addWeighted(
    swir,
    0.5,
    warped_vnir,
    0.5,
    0
)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(vnir, cmap="gray")
axes[0].set_title("Original VNIR reference")

axes[1].imshow(swir, cmap="gray")
axes[1].set_title("Original SWIR reference")

axes[2].imshow(overlay, cmap="gray")
axes[2].set_title("VNIR warped onto SWIR")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output = OUT_DIR / f"{SAMPLE}_registration_diagnostic.png"
plt.savefig(output, dpi=150)
plt.close()

print("\nDiagnostic saved:", output)
print("\nREGISTRATION INVESTIGATION COMPLETED")
