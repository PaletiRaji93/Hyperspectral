
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import spectral


# ============================================================
# TERRASPECTRA - 3D CNN PATCH PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    ROOT / "artifacts/dataset_inspection/master_sample_metadata.csv"
)

VNIR_DIR = ROOT / "dataset/raw/VNIR"

OUTPUT_DIR = ROOT / "outputs/3dcnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 11
RADIUS = PATCH_SIZE // 2

MAX_PATCHES_PER_LEAF = 12
MIN_MASK_FRACTION = 0.70

IGNORE_VALUE = 4095
RANDOM_SEED = 42


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def select_patch_centers(coords, height, width, rng):
    """Select spatially separated centers inside annotated leaf regions."""

    rows = coords[:, 0]
    cols = coords[:, 1]

    # Keep centers away from cube borders.
    valid = (
        (rows >= RADIUS)
        & (rows < height - RADIUS)
        & (cols >= RADIUS)
        & (cols < width - RADIUS)
    )

    rows = rows[valid]
    cols = cols[valid]

    if len(rows) == 0:
        return [], []

    # Build a binary leaf mask.
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[coords[:, 0], coords[:, 1]] = 1

    # Integral image for fast patch occupancy calculation.
    integral = np.pad(
    mask.astype(np.int32),
    ((1, 0), (1, 0)),
    mode="constant"
).cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)

    r0 = rows - RADIUS
    r1 = rows + RADIUS + 1
    c0 = cols - RADIUS
    c1 = cols + RADIUS + 1

    counts = (
        integral[r1, c1]
        - integral[r0, c1]
        - integral[r1, c0]
        + integral[r0, c0]
    )

    minimum_pixels = int(
        np.ceil(PATCH_SIZE * PATCH_SIZE * MIN_MASK_FRACTION)
    )

    good = counts >= minimum_pixels

    rows = rows[good]
    cols = cols[good]
    counts = counts[good]

    if len(rows) == 0:
        return [], []

    # Randomized greedy selection to reduce patch overlap.
    order = rng.permutation(len(rows))

    selected = []

    for idx in order:
        r = int(rows[idx])
        c = int(cols[idx])

        separated = all(
            max(abs(r - old_r), abs(c - old_c)) >= PATCH_SIZE
            for old_r, old_c in selected
        )

        if separated:
            selected.append((r, c))

        if len(selected) >= MAX_PATCHES_PER_LEAF:
            break

    fractions = []

    for r, c in selected:
        rr = r
        cc = c

        # Look up the annotated-pixel fraction in this patch.
        count = (
            integral[rr + RADIUS + 1, cc + RADIUS + 1]
            - integral[rr - RADIUS, cc + RADIUS + 1]
            - integral[rr + RADIUS + 1, cc - RADIUS]
            + integral[rr - RADIUS, cc - RADIUS]
        )

        fractions.append(
            count / float(PATCH_SIZE * PATCH_SIZE)
        )

    return selected, fractions


def extract_patch(cube, row, col):
    """Extract one patch and replace declared ignore values."""

    patch = np.asarray(
        cube[
            row - RADIUS:row + RADIUS + 1,
            col - RADIUS:col + RADIUS + 1,
            :
        ],
        dtype=np.float32
    )

    if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError("Unexpected patch dimensions")

    # Replace VNIR ignore values with the patch's valid
    # per-band median. This is local imputation, not calibration.
    for band in range(patch.shape[2]):
        band_data = patch[:, :, band]

        invalid = band_data == IGNORE_VALUE
        valid_values = band_data[~invalid]

        if np.any(invalid):
            replacement = (
                np.median(valid_values)
                if len(valid_values) > 0
                else 0.0
            )

            band_data[invalid] = replacement

    # Return layout: spectral bands, height, width.
    return np.rint(
        np.clip(patch.transpose(2, 0, 1), 0, 65535)
    ).astype(np.uint16)


def main():

    print("=" * 70)
    print("TERRASPECTRA - 3D CNN PATCH PREPARATION")
    print("=" * 70)

    metadata = pd.read_csv(METADATA_PATH)

    metadata = metadata[
        metadata["Eligible_For_Leaf_Analysis"].apply(to_bool)
        & metadata["VNIR_Leaf_Annotation"].apply(to_bool)
    ].copy()

    treatment_mapping = {
        "Non-infected": 0,
        "Infected": 1
    }

    metadata["Disease_Label"] = metadata["Treatment"].map(
        treatment_mapping
    )

    if metadata["Disease_Label"].isna().any():
        raise ValueError("Unknown disease labels found.")

    metadata["Disease_Label"] = (
        metadata["Disease_Label"].astype(int)
    )

    # Confirm that every pot has one disease label.
    pot_labels = metadata.groupby("Pot")["Disease_Label"].nunique()

    if (pot_labels > 1).any():
        raise ValueError("A pot has multiple disease labels.")

    print("\nEligible leaves:", len(metadata))
    print("Unique pots:", metadata["Pot"].nunique())
    print("\nDisease distribution:")
    print(metadata["Treatment"].value_counts())

    rng = np.random.default_rng(RANDOM_SEED)

    all_patches = []
    patch_records = []
    leaves_without_patches = []

    sample_groups = metadata.groupby("Sample_ID")

    for sample_number, (sample_id, group) in enumerate(
        sample_groups, start=1
    ):

        sample_folder = VNIR_DIR / str(sample_id)
        header_path = sample_folder / f"{sample_id}.hdr"

        if not header_path.exists():
            raise FileNotFoundError(header_path)

        image = spectral.open_image(str(header_path))
        cube = image.open_memmap()

        if cube.ndim != 3 or cube.shape[2] != 216:
            raise ValueError(
                f"Unexpected cube shape for {sample_id}: {cube.shape}"
            )

        for _, leaf in group.iterrows():

            leaf_number = int(leaf["Leaf"])

            annotation_path = (
                sample_folder
                / f"{sample_id}_leaf{leaf_number}.csv"
            )

            if not annotation_path.exists():
                raise FileNotFoundError(annotation_path)

            coords = pd.read_csv(
                annotation_path,
                header=None
            ).iloc[:, :2].to_numpy(dtype=np.int64)

            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("Invalid annotation coordinates")

            coords = np.unique(coords, axis=0)

            if (
                np.any(coords[:, 0] < 0)
                or np.any(coords[:, 0] >= cube.shape[0])
                or np.any(coords[:, 1] < 0)
                or np.any(coords[:, 1] >= cube.shape[1])
            ):
                raise ValueError(
                    f"Coordinates outside cube: {sample_id}, leaf {leaf_number}"
                )

            centers, fractions = select_patch_centers(
                coords,
                cube.shape[0],
                cube.shape[1],
                rng
            )

            if not centers:
                leaves_without_patches.append(
                    f"{sample_id}_leaf{leaf_number}"
                )
                continue

            for patch_number, ((row, col), fraction) in enumerate(
                zip(centers, fractions), start=1
            ):

                patch = extract_patch(cube, row, col)

                all_patches.append(patch)

                patch_records.append({
                    "Patch_ID": (
                        f"{sample_id}_L{leaf_number}_P{patch_number}"
                    ),
                    "Sample_ID": sample_id,
                    "Pot": leaf["Pot"],
                    "Leaf": leaf_number,
                    "Treatment": leaf["Treatment"],
                    "Disease_Label": int(leaf["Disease_Label"]),
                    "Genotype": leaf["Genotype"],
                    "Acquisition_Date": leaf["Acquisition_Date"],
                    "Center_Row": row,
                    "Center_Col": col,
                    "Mask_Fraction": round(fraction, 4)
                })

        del cube
        del image
        gc.collect()

        if sample_number % 10 == 0:
            print(
                f"Processed {sample_number}/"
                f"{len(sample_groups)} sample folders"
            )

    if not all_patches:
        raise RuntimeError("No valid patches were generated.")

    patch_array = np.stack(all_patches)

    patch_index = pd.DataFrame(patch_records)

    patch_path = OUTPUT_DIR / "vnir_patches.npy"
    index_path = OUTPUT_DIR / "vnir_patch_index.csv"

    np.save(patch_path, patch_array)
    patch_index.to_csv(index_path, index=False)

    print("\n" + "=" * 70)
    print("PATCH PREPARATION SUMMARY")
    print("=" * 70)

    print("Patch array shape:", patch_array.shape)
    print("Data type:", patch_array.dtype)
    print("Patch size:", PATCH_SIZE)
    print("Spectral bands:", patch_array.shape[1])

    print("\nPatch counts by disease:")
    print(patch_index.groupby("Treatment").size())

    print("\nUnique pots by disease:")
    print(
        patch_index.groupby("Treatment")["Pot"].nunique()
    )

    print("\nLeaves represented:", patch_index[
        ["Sample_ID", "Leaf"]
    ].drop_duplicates().shape[0])

    print("Leaves without valid patches:", len(leaves_without_patches))

    if leaves_without_patches:
        print("\nLeaves requiring investigation:")
        print(leaves_without_patches)

    print("\nSaved patch array:", patch_path)
    print("Saved patch index:", index_path)

    print("\nOriginal hyperspectral cubes were not modified.")
    print("PATCH PREPARATION FINISHED")


if __name__ == "__main__":
    main()
