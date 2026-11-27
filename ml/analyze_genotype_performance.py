
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "outputs" / "ml_datasets"
    / "multisensor_leaf_ml_dataset.csv"
)

PREDICTION_PATH = (
    ROOT / "artifacts" / "baseline_model_evaluation"
    / "baseline_oof_predictions.csv"
)

OUTPUT_DIR = ROOT / "artifacts" / "genotype_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(group):

    y_true = group["Actual_Label"].astype(int)
    y_pred = group["Predicted_Label"].astype(int)
    scores = group["Prediction_Score"].astype(float)

    result = {
        "Records": len(group),
        "Infected": int((y_true == 1).sum()),
        "Non_Infected": int((y_true == 0).sum()),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }

    if y_true.nunique() == 2:
        result["Balanced_Accuracy"] = balanced_accuracy_score(
            y_true, y_pred
        )
        result["ROC_AUC"] = roc_auc_score(
            y_true, scores
        )
    else:
        result["Balanced_Accuracy"] = np.nan
        result["ROC_AUC"] = np.nan

    return result


def main():

    print("=" * 75)
    print("TERRASPECTRA - GENOTYPE PERFORMANCE ANALYSIS")
    print("=" * 75)

    data = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)

    # Select combined SVM out-of-fold predictions.
    predictions = predictions[
        (predictions["Dataset"] == "Combined")
        & (predictions["Model"] == "SVM_RBF")
    ].copy()

    if predictions.empty:
        raise ValueError("Combined SVM predictions not found.")

    # Recover genotype and treatment metadata using original row index.
    metadata = data[
        ["Genotype", "Treatment", "Disease_Label", "Pot"]
    ].copy()

    metadata["Original_Row"] = np.arange(len(metadata))

    results = predictions.merge(
        metadata,
        on="Original_Row",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", "_metadata")
    )

    if results["Genotype"].isna().any():
        raise ValueError("Some predictions could not be matched to genotype.")

    if len(results) != len(data):
        raise ValueError(
            "Expected exactly one out-of-fold prediction per leaf."
        )

    if not (
        results["Actual_Label"].to_numpy()
        == results["Disease_Label"].to_numpy()
    ).all():
        raise ValueError("Prediction labels do not match source dataset.")

    print("\n1. DATASET DISTRIBUTION")
    print("-" * 55)

    distribution = pd.crosstab(
        results["Genotype"],
        results["Treatment"],
        margins=True
    )

    print(distribution)

    print("\n2. PERFORMANCE BY GENOTYPE")
    print("-" * 55)

    genotype_results = []

    for genotype, group in results.groupby("Genotype"):

        metrics = calculate_metrics(group)
        metrics["Genotype"] = genotype

        genotype_results.append(metrics)

        print(f"\nGenotype: {genotype}")

        for key, value in metrics.items():
            if key == "Genotype":
                continue

            if isinstance(value, (float, np.floating)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        cm = confusion_matrix(
            group["Actual_Label"],
            group["Predicted_Label"],
            labels=[0, 1]
        )

        print("\nConfusion matrix:")
        print("Rows = Actual [Non-infected, Infected]")
        print("Columns = Predicted [Non-infected, Infected]")
        print(cm)

    genotype_df = pd.DataFrame(genotype_results)

    genotype_path = OUTPUT_DIR / "genotype_performance.csv"
    genotype_df.to_csv(genotype_path, index=False)

    print("\n3. PERFORMANCE BY GENOTYPE AND FOLD")
    print("-" * 55)

    fold_results = []

    for (genotype, fold), group in results.groupby(
        ["Genotype", "Fold"]
    ):

        metrics = calculate_metrics(group)
        metrics["Genotype"] = genotype
        metrics["Fold"] = fold

        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)

    print(
        fold_df[
            [
                "Genotype",
                "Fold",
                "Records",
                "Balanced_Accuracy",
                "F1",
                "ROC_AUC",
            ]
        ].round(4).to_string(index=False)
    )

    fold_path = OUTPUT_DIR / "genotype_fold_performance.csv"
    fold_df.to_csv(fold_path, index=False)

    print("\n4. GENOTYPE LABEL DISTRIBUTION")
    print("-" * 55)

    label_distribution = (
        results.groupby(["Genotype", "Disease_Label"])
        .size()
        .unstack(fill_value=0)
    )

    print(label_distribution)

    distribution_path = (
        OUTPUT_DIR / "genotype_label_distribution.csv"
    )
    label_distribution.to_csv(distribution_path)

    print("\nSaved reports:")
    print(genotype_path)
    print(fold_path)
    print(distribution_path)

    print("\n" + "=" * 75)
    print("GENOTYPE ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
