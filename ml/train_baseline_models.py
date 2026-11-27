
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "outputs" / "ml_datasets"
OUTPUT_DIR = ROOT / "artifacts" / "baseline_model_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "VNIR": DATA_DIR / "vnir_ml_dataset.csv",
    "SWIR": DATA_DIR / "swir_ml_dataset.csv",
    "Combined": DATA_DIR / "multisensor_leaf_ml_dataset.csv",
}

TARGET = "Disease_Label"
GROUP = "Pot"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_mean_features(df, dataset_name):

    if dataset_name == "Combined":
        features = [
            col for col in df.columns
            if col.startswith("VNIR_mean_")
            or col.startswith("SWIR_mean_")
        ]
    else:
        features = [
            col for col in df.columns
            if col.startswith("mean_")
        ]

    if not features:
        raise ValueError(
            f"No mean spectral features found for {dataset_name}"
        )

    return features


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():

    return {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            ))
        ]),

        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random_Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_score):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, y_score),
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_dataset(dataset_name, file_path):

    print("\n" + "=" * 75)
    print(f"DATASET: {dataset_name}")
    print("=" * 75)

    df = pd.read_csv(file_path)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if GROUP not in df.columns:
        raise ValueError(f"Missing grouping column: {GROUP}")

    features = get_mean_features(df, dataset_name)

    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)
    groups = df[GROUP]

    if X.isna().any().any():
        raise ValueError(
            f"{dataset_name}: missing or non-numeric feature values found."
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{dataset_name}: infinite feature values found."
        )

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"{dataset_name}: expected binary labels 0 and 1."
        )

    if df.duplicated(["Sample_ID", "Leaf"]).any():
        raise ValueError(
            f"{dataset_name}: duplicate sample/leaf records found."
        )

    # Every pot must contain only one disease label.
    pot_label_counts = df.groupby(GROUP)[TARGET].nunique()

    if (pot_label_counts > 1).any():
        raise ValueError(
            f"{dataset_name}: a pot contains conflicting disease labels."
        )

    print(f"Records: {len(df)}")
    print(f"Number of features: {len(features)}")
    print(f"Unique pots: {groups.nunique()}")
    print(f"Infected: {(y == 1).sum()}")
    print(f"Non-infected: {(y == 0).sum()}")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = create_models()

    fold_results = []
    prediction_results = []

    for model_name, model in models.items():

        print(f"\nModel: {model_name}")

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])

            # Explicitly verify no pot leakage.
            if train_groups.intersection(test_groups):
                raise RuntimeError(
                    f"Group leakage detected in fold {fold}!"
                )

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                raise ValueError(
                    f"Fold {fold} does not contain both classes. "
                    "Review group distribution or reduce the number of folds."
                )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.decision_function(X_test)

            metrics = calculate_metrics(
                y_test,
                predictions,
                scores
            )

            metrics.update({
                "Dataset": dataset_name,
                "Model": model_name,
                "Fold": fold,
                "Train_Records": len(train_idx),
                "Test_Records": len(test_idx),
                "Train_Pots": len(train_groups),
                "Test_Pots": len(test_groups),
            })

            fold_results.append(metrics)

            for position, original_idx in enumerate(test_idx):

                prediction_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Original_Row": int(original_idx),
                    "Sample_ID": df.iloc[original_idx]["Sample_ID"],
                    "Pot": df.iloc[original_idx]["Pot"],
                    "Leaf": df.iloc[original_idx]["Leaf"],
                    "Actual_Label": int(y_test.iloc[position]),
                    "Predicted_Label": int(predictions[position]),
                    "Prediction_Score": float(scores[position]),
                    "Fold": fold,
                })

            print(
                f"  Fold {fold}: "
                f"Balanced Accuracy={metrics['Balanced_Accuracy']:.4f}, "
                f"F1={metrics['F1_Score']:.4f}, "
                f"ROC-AUC={metrics['ROC_AUC']:.4f}"
            )

    return fold_results, prediction_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TERRASPECTRA - BASELINE ML MODEL EVALUATION")
    print("=" * 75)

    print("\nValidation strategy:")
    print("5-fold Stratified Group Cross-Validation")
    print("Grouping variable: Pot")
    print("Features: Mean spectral values only")
    print("No individual-pot leakage between train and test.")

    all_fold_results = []
    all_predictions = []

    for dataset_name, file_path in DATASETS.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        fold_results, predictions = evaluate_dataset(
            dataset_name,
            file_path
        )

        all_fold_results.extend(fold_results)
        all_predictions.extend(predictions)

    # --------------------------------------------------------
    # SAVE FOLD RESULTS
    # --------------------------------------------------------

    fold_df = pd.DataFrame(all_fold_results)

    fold_path = OUTPUT_DIR / "baseline_fold_metrics.csv"
    fold_df.to_csv(fold_path, index=False)

    # --------------------------------------------------------
    # SAVE OUT-OF-FOLD PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(all_predictions)

    prediction_path = OUTPUT_DIR / "baseline_oof_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    # --------------------------------------------------------
    # SUMMARY: MEAN AND STANDARD DEVIATION
    # --------------------------------------------------------

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
    ]

    summary = (
        fold_df.groupby(["Dataset", "Model"])[metric_columns]
        .agg(["mean", "std"])
        .round(4)
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]

    summary = summary.reset_index()

    summary_path = OUTPUT_DIR / "baseline_model_summary.csv"
    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # OUT-OF-FOLD CONFUSION MATRICES
    # --------------------------------------------------------

    for (dataset_name, model_name), group in prediction_df.groupby(
        ["Dataset", "Model"]
    ):

        y_true = group["Actual_Label"]
        y_pred = group["Predicted_Label"]

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        image = ax.imshow(cm)

        ax.set_title(f"{dataset_name} - {model_name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-infected", "Infected"])
        ax.set_yticklabels(["Non-infected", "Infected"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        filename = (
            f"{dataset_name}_{model_name}_confusion_matrix.png"
        )

        fig.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close(fig)

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("BASELINE MODEL COMPARISON")
    print("=" * 75)

    display_columns = [
        "Dataset",
        "Model",
        "Balanced_Accuracy_mean",
        "Balanced_Accuracy_std",
        "F1_Score_mean",
        "F1_Score_std",
        "ROC_AUC_mean",
        "ROC_AUC_std",
    ]

    print(
        summary[display_columns]
        .sort_values(
            "Balanced_Accuracy_mean",
            ascending=False
        )
        .to_string(index=False)
    )

    print("\nSaved results:")
    print(f"Fold metrics: {fold_path}")
    print(f"Summary: {summary_path}")
    print(f"Out-of-fold predictions: {prediction_path}")
    print(f"Confusion matrices: {OUTPUT_DIR}")

    print("\nIMPORTANT:")
    print("These are baseline cross-validation results,")
    print("not proof of generalization to new farms, dates,")
    print("genotypes, or independent experiments.")

    print("\n" + "=" * 75)
    print("BASELINE MODEL EVALUATION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
