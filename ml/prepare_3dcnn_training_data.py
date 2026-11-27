
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# TERRASPECTRA - 3D CNN TRAINING DATA PREPARATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PATCH_PATH = ROOT / "outputs/3dcnn/vnir_patches.npy"
INDEX_PATH = ROOT / "outputs/3dcnn/vnir_patch_index.csv"

OUTPUT_DIR = ROOT / "outputs/3dcnn/training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Split proportions: 70% train, 15% validation, 15% test
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def main():

    print("=" * 70)
    print("TERRASPECTRA - 3D CNN TRAINING DATA PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD PATCH DATA AND METADATA
    # --------------------------------------------------------

    X = np.load(PATCH_PATH)
    index = pd.read_csv(INDEX_PATH)

    print("\n1. LOADED DATA")
    print("Patch array:", X.shape)
    print("Patch index:", index.shape)

    if len(X) != len(index):
        raise ValueError("Patch array and index row counts do not match.")

    if not np.isfinite(X).all():
        raise ValueError("Patch data contains NaN or infinite values.")

    required_columns = [
        "Patch_ID", "Pot", "Treatment", "Disease_Label"
    ]

    missing = [
        col for col in required_columns
        if col not in index.columns
    ]

    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    # --------------------------------------------------------
    # 2. VERIFY POT-LEVEL LABEL CONSISTENCY
    # --------------------------------------------------------

    pot_labels = index.groupby("Pot")["Disease_Label"].nunique()

    if (pot_labels != 1).any():
        raise ValueError("Some pots contain multiple disease labels.")

    pot_table = (
        index[["Pot", "Disease_Label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("\n2. POT-LEVEL DATA")
    print("Unique pots:", len(pot_table))
    print(pot_table["Disease_Label"].value_counts().sort_index())

    # --------------------------------------------------------
    # 3. SPLIT POTS: 70% / 15% / 15%
    # --------------------------------------------------------

    train_pots, temp_pots = train_test_split(
        pot_table,
        test_size=(1 - TRAIN_SIZE),
        stratify=pot_table["Disease_Label"],
        random_state=RANDOM_SEED
    )

    validation_fraction = VALIDATION_SIZE / (
        VALIDATION_SIZE + TEST_SIZE
    )

    val_pots, test_pots = train_test_split(
        temp_pots,
        test_size=(1 - validation_fraction),
        stratify=temp_pots["Disease_Label"],
        random_state=RANDOM_SEED
    )

    train_ids = set(train_pots["Pot"])
    val_ids = set(val_pots["Pot"])
    test_ids = set(test_pots["Pot"])

    # Verify no pot leakage
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    train_mask = index["Pot"].isin(train_ids).to_numpy()
    val_mask = index["Pot"].isin(val_ids).to_numpy()
    test_mask = index["Pot"].isin(test_ids).to_numpy()

    print("\n3. SPLIT SUMMARY")

    for name, mask in [
        ("Train", train_mask),
        ("Validation", val_mask),
        ("Test", test_mask)
    ]:
        subset = index.loc[mask]

        print(f"\n{name}:")
        print("  Patches:", len(subset))
        print("  Pots:", subset["Pot"].nunique())
        print("  Leaves:", subset["Sample_ID"].astype(str).nunique())
        print("  Disease distribution:")
        print(subset["Treatment"].value_counts())

    # --------------------------------------------------------
    # 4. NORMALIZATION USING TRAINING DATA ONLY
    # --------------------------------------------------------

    print("\n4. CALCULATING TRAINING NORMALIZATION")

    # Calculate per-spectral-band statistics.
    # Patch layout: (N, Bands, Height, Width)

    X_float = X.astype(np.float32)

    train_data = X_float[train_mask]

    mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
    std = train_data.std(axis=(0, 2, 3), keepdims=True)

    # Prevent division by zero
    std[std < 1e-6] = 1.0

    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)

    # Normalize using training statistics
    X_normalized = (X_float - mean) / std

    # --------------------------------------------------------
    # 5. CREATE LABELS
    # --------------------------------------------------------

    y = index["Disease_Label"].to_numpy(dtype=np.int64)

    # --------------------------------------------------------
    # 6. SAVE SPLITS
    # --------------------------------------------------------

    np.save(OUTPUT_DIR / "X_train.npy", X_normalized[train_mask])
    np.save(OUTPUT_DIR / "X_val.npy", X_normalized[val_mask])
    np.save(OUTPUT_DIR / "X_test.npy", X_normalized[test_mask])

    np.save(OUTPUT_DIR / "y_train.npy", y[train_mask])
    np.save(OUTPUT_DIR / "y_val.npy", y[val_mask])
    np.save(OUTPUT_DIR / "y_test.npy", y[test_mask])

    # Save normalization statistics
    np.savez(
        OUTPUT_DIR / "normalization_stats.npz",
        mean=mean,
        std=std
    )

    # Save patch metadata with split assignments
    index = index.copy()
    index["Split"] = "train"
    index.loc[val_mask, "Split"] = "validation"
    index.loc[test_mask, "Split"] = "test"

    index.to_csv(
        OUTPUT_DIR / "patch_split_index.csv",
        index=False
    )

    # Save pot-level split information
    split_info = {
        "random_seed": RANDOM_SEED,
        "train_pots": sorted(map(str, train_ids)),
        "validation_pots": sorted(map(str, val_ids)),
        "test_pots": sorted(map(str, test_ids)),
        "split_strategy": "Stratified pot-level split",
        "normalization": "Per-band mean and standard deviation from training patches only",
        "patch_shape": list(X.shape[1:])
    }

    with open(OUTPUT_DIR / "split_config.json", "w") as f:
        json.dump(split_info, f, indent=4)

    # --------------------------------------------------------
    # 7. FINAL VERIFICATION
    # --------------------------------------------------------

    print("\n5. FINAL DATASET SHAPES")

    for name in ["train", "val", "test"]:
        features = np.load(
            OUTPUT_DIR / f"X_{name}.npy",
            mmap_mode="r"
        )
        labels = np.load(OUTPUT_DIR / f"y_{name}.npy")

        print(f"{name.upper()}:")
        print("  X:", features.shape)
        print("  y:", labels.shape)
        print("  Label counts:", np.bincount(labels, minlength=2))
        print("  Finite values:", np.isfinite(features).all())

    print("\nPot leakage check: PASSED")
    print("Normalization source: Training patches only")

    print("\nSaved files in:")
    print(OUTPUT_DIR)

    print("\nTRAINING DATA PREPARATION FINISHED")


if __name__ == "__main__":
    main()
