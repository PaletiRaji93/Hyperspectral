
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "outputs" / "ml_datasets"
OUTPUT_DIR = ROOT / "artifacts" / "pca_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "VNIR": DATA_DIR / "vnir_ml_dataset.csv",
    "SWIR": DATA_DIR / "swir_ml_dataset.csv",
    "Combined": DATA_DIR / "multisensor_leaf_ml_dataset.csv",
}

VARIANCE_LEVELS = [None, 0.90, 0.95, 0.99]
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(df, dataset_name):

    if dataset_name == "Combined":
        columns = [
            c for c in df.columns
            if c.startswith("VNIR_mean_")
            or c.startswith("SWIR_mean_")
        ]
    else:
        columns = [
            c for c in df.columns
            if c.startswith("mean_")
        ]

    if not columns:
        raise ValueError(
            f"No spectral mean features found: {dataset_name}"
        )

    return columns


# ============================================================
# MODEL
# ============================================================

def build_model(variance):

    steps = [
        ("scaler", StandardScaler())
    ]

    if variance is not None:
        steps.append(
            ("pca", PCA(
                n_components=variance,
                svd_solver="full"
            ))
        )

    steps.append(
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    )

    return Pipeline(steps)


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(dataset_name, path):

    print("\n" + "=" * 70)
    print(f"PCA EXPERIMENT: {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(path)

    feature_columns = select_features(df, dataset_name)

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = df["Disease_Label"].astype(int)
    groups = df["Pot"]

    if X.isna().any().any():
        raise ValueError("Missing feature values detected.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Infinite feature values detected.")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    for variance in VARIANCE_LEVELS:

        label = (
            "Original"
            if variance is None
            else f"PCA_{int(variance * 100)}"
        )

        print(f"\nExperiment: {label}")

        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])

            if train_groups.intersection(test_groups):
                raise RuntimeError("Pot leakage detected.")

            model = build_model(variance)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            scores = model.decision_function(X_test)

            balanced_acc = balanced_accuracy_score(
                y_test, predictions
            )

            f1 = f1_score(
                y_test, predictions, zero_division=0
            )

            auc = roc_auc_score(y_test, scores)

            if variance is None:
                component_count = len(feature_columns)
                explained_variance = 1.0
            else:
                pca = model.named_steps["pca"]
                component_count = pca.n_components_
                explained_variance = (
                    pca.explained_variance_ratio_.sum()
                )

            fold_scores.append({
                "Dataset": dataset_name,
                "Experiment": label,
                "Fold": fold,
                "Balanced_Accuracy": balanced_acc,
                "F1_Score": f1,
                "ROC_AUC": auc,
                "Components": component_count,
                "Explained_Variance": explained_variance,
            })

            print(
                f"Fold {fold}: "
                f"Balanced Accuracy={balanced_acc:.4f}, "
                f"F1={f1:.4f}, "
                f"ROC-AUC={auc:.4f}, "
                f"Components={component_count}"
            )

        results.extend(fold_scores)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TERRASPECTRA - PCA + SVM EXPERIMENT")
    print("=" * 70)

    all_results = []

    for name, path in DATASETS.items():

        if not path.exists():
            raise FileNotFoundError(path)

        all_results.extend(
            run_experiment(name, path)
        )

    fold_df = pd.DataFrame(all_results)

    fold_path = OUTPUT_DIR / "pca_fold_metrics.csv"
    fold_df.to_csv(fold_path, index=False)

    summary = (
        fold_df.groupby(["Dataset", "Experiment"])
        .agg(
            Balanced_Accuracy_Mean=(
                "Balanced_Accuracy", "mean"
            ),
            Balanced_Accuracy_Std=(
                "Balanced_Accuracy", "std"
            ),
            F1_Mean=("F1_Score", "mean"),
            F1_Std=("F1_Score", "std"),
            ROC_AUC_Mean=("ROC_AUC", "mean"),
            ROC_AUC_Std=("ROC_AUC", "std"),
            Average_Components=("Components", "mean"),
            Average_Explained_Variance=(
                "Explained_Variance", "mean"
            ),
        )
        .round(4)
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "pca_model_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("PCA EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        summary.sort_values(
            "Balanced_Accuracy_Mean",
            ascending=False
        ).to_string(index=False)
    )

    print("\nSaved:")
    print(f"Fold results: {fold_path}")
    print(f"Summary: {summary_path}")

    print("\n" + "=" * 70)
    print("PCA EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
