
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# ============================================================
# TERRASPECTRA - LEAF AND POT LEVEL 3D CNN EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = ROOT / "outputs/3dcnn/training/patch_split_index.csv"
PREDICTIONS_PATH = ROOT / "outputs/3dcnn/model/test_patch_predictions.csv"

OUTPUT_DIR = ROOT / "outputs/3dcnn/model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_level(data, level_name):

    y_true = data["Disease_Label"].to_numpy(dtype=int)
    y_prob = data["Infected_Probability"].to_numpy(dtype=float)

    y_pred = (y_prob >= 0.5).astype(int)

    print("\n" + "=" * 70)
    print(f"{level_name.upper()} LEVEL EVALUATION")
    print("=" * 70)

    print("Number of observations:", len(data))
    print("Class distribution:")
    print(data["Disease_Label"].value_counts().sort_index())

    print("\nAccuracy:",
          f"{accuracy_score(y_true, y_pred):.4f}")

    print("Balanced accuracy:",
          f"{balanced_accuracy_score(y_true, y_pred):.4f}")

    print("F1-score:",
          f"{f1_score(y_true, y_pred, zero_division=0):.4f}")

    if len(np.unique(y_true)) == 2:
        print("ROC-AUC:",
              f"{roc_auc_score(y_true, y_prob):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Non-infected", "Infected"],
            zero_division=0
        )
    )

    return {
        "observations": len(data),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2
            else np.nan
        )
    }


def main():

    print("=" * 70)
    print("TERRASPECTRA - 3D CNN LEAF/POT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD TEST METADATA AND PATCH PREDICTIONS
    # --------------------------------------------------------

    index = pd.read_csv(INDEX_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    test_index = (
        index[index["Split"] == "test"]
        .reset_index(drop=True)
    )

    if len(test_index) != len(predictions):
        raise ValueError(
            "Test metadata and prediction counts do not match."
        )

    required = [
        "Sample_ID", "Leaf", "Pot",
        "Treatment", "Disease_Label"
    ]

    missing = [c for c in required if c not in test_index.columns]

    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    # Confirm predictions correspond to test patches in saved order
    if not np.array_equal(
        predictions["y_true"].to_numpy(),
        test_index["Disease_Label"].to_numpy()
    ):
        raise ValueError(
            "Prediction labels do not match test metadata order."
        )

    test_index["Infected_Probability"] = (
        predictions["infected_probability"].to_numpy()
    )

    print("\n1. TEST PATCHES")
    print("Patches:", len(test_index))
    print("Unique pots:", test_index["Pot"].nunique())
    print(
        "Unique leaves:",
        test_index[["Sample_ID", "Leaf"]].drop_duplicates().shape[0]
    )

    # --------------------------------------------------------
    # 2. PATCH -> LEAF AGGREGATION
    # --------------------------------------------------------

    # Each leaf receives the mean infected probability
    # from its patches. Each leaf contributes one observation.

    leaf_data = (
        test_index.groupby(
            ["Sample_ID", "Leaf", "Pot"],
            as_index=False
        )
        .agg(
            Disease_Label=("Disease_Label", "first"),
            Treatment=("Treatment", "first"),
            Infected_Probability=("Infected_Probability", "mean"),
            Patch_Count=("Infected_Probability", "size")
        )
    )

    # Confirm each leaf has a consistent label
    label_counts = (
        test_index.groupby(["Sample_ID", "Leaf"])["Disease_Label"]
        .nunique()
    )

    if (label_counts != 1).any():
        raise ValueError("A leaf has inconsistent disease labels.")

    print("\n2. LEAF AGGREGATION")
    print("Leaves:", len(leaf_data))
    print("Patches per leaf:")
    print(leaf_data["Patch_Count"].value_counts().sort_index())

    leaf_metrics = evaluate_level(leaf_data, "LEAF")

    leaf_data.to_csv(
        OUTPUT_DIR / "test_leaf_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # 3. LEAF -> POT AGGREGATION
    # --------------------------------------------------------

    # First average patch probabilities within each leaf.
    # Then average leaf probabilities within each pot.
    # This gives each leaf equal weight within its pot.

    pot_data = (
        leaf_data.groupby("Pot", as_index=False)
        .agg(
            Disease_Label=("Disease_Label", "first"),
            Treatment=("Treatment", "first"),
            Infected_Probability=("Infected_Probability", "mean"),
            Leaf_Count=("Leaf", "count")
        )
    )

    pot_label_counts = (
        leaf_data.groupby("Pot")["Disease_Label"].nunique()
    )

    if (pot_label_counts != 1).any():
        raise ValueError("A pot has inconsistent disease labels.")

    print("\n3. POT AGGREGATION")
    print("Pots:", len(pot_data))

    pot_metrics = evaluate_level(pot_data, "POT")

    pot_data.to_csv(
        OUTPUT_DIR / "test_pot_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # 4. SAVE COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {"Level": "Patch", **{
                "observations": len(test_index),
                "accuracy": accuracy_score(
                    predictions["y_true"],
                    predictions["y_pred"]
                ),
                "balanced_accuracy": np.nan,
                "f1": np.nan,
                "auc": roc_auc_score(
                    predictions["y_true"],
                    predictions["infected_probability"]
                )
            }},
            {"Level": "Leaf", **leaf_metrics},
            {"Level": "Pot", **pot_metrics}
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR / "level_comparison.csv",
        index=False
    )

    print("\n4. COMPARISON")
    print(comparison.to_string(index=False))

    print("\nSaved evaluation files:")
    print(" - test_leaf_predictions.csv")
    print(" - test_pot_predictions.csv")
    print(" - level_comparison.csv")

    print("\nEVALUATION FINISHED")


if __name__ == "__main__":
    main()
