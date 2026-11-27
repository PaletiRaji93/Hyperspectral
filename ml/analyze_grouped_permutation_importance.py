import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "outputs/ml_datasets/multisensor_leaf_ml_dataset.csv"

OUTPUT_DIR = "artifacts/permutation_importance"

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("TERRASPECTRA - GROUPED PERMUTATION FEATURE IMPORTANCE")
print("=" * 75)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

# Use treatment as the disease target.
# Infected = 1, Non-infected = 0.
target = (
    df["Treatment"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "infected": 1,
        "non-infected": 0,
        "non infected": 0
    })
)

if target.isna().any():
    raise ValueError(
        "Unknown or missing Treatment values. "
        f"Found: {df['Treatment'].unique()}"
    )

y = target.astype(int)

# Pot is the grouping variable to prevent pot-level leakage.
groups = df["Pot"].astype(str)

# Use mean spectral features only, matching the baseline experiment.
feature_columns = [
    col for col in df.columns
    if col.startswith("VNIR_mean_") or col.startswith("SWIR_mean_")
]

if not feature_columns:
    raise ValueError("No VNIR_mean_ or SWIR_mean_ features found.")

X = df[feature_columns].copy()

# Ensure numeric values.
X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError("Missing or non-numeric values found in spectral features.")

print("\nFEATURE INFORMATION")
print("-" * 50)
print("Records:", len(X))
print("Spectral features:", X.shape[1])
print("VNIR features:", sum(c.startswith("VNIR_") for c in feature_columns))
print("SWIR features:", sum(c.startswith("SWIR_") for c in feature_columns))
print("Unique pots:", groups.nunique())
print("Infected:", int((y == 1).sum()))
print("Non-infected:", int((y == 0).sum()))


# ============================================================
# MODEL AND GROUPED CROSS-VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    ))
])

fold_results = []
importance_results = []

print("\nCROSS-VALIDATED PERMUTATION IMPORTANCE")
print("-" * 75)

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

    # Safety check: no pot may occur in both sets.
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Pot leakage detected in fold {fold}: {overlap}"
        )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    baseline_score = balanced_accuracy_score(
        y_test,
        predictions
    )

    print(f"\nFold {fold}/{N_SPLITS}")
    print("Training records:", len(train_idx))
    print("Testing records:", len(test_idx))
    print("Training pots:", len(train_groups))
    print("Testing pots:", len(test_groups))
    print(f"Baseline balanced accuracy: {baseline_score:.4f}")

    # Permute one feature at a time in held-out data.
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE + fold,
        n_jobs=1
    )

    for i, feature in enumerate(feature_columns):
        importance_results.append({
            "Fold": fold,
            "Feature": feature,
            "Importance_Mean": result.importances_mean[i],
            "Importance_Std": result.importances_std[i],
            "Baseline_Balanced_Accuracy": baseline_score
        })

    fold_results.append({
        "Fold": fold,
        "Train_Records": len(train_idx),
        "Test_Records": len(test_idx),
        "Train_Pots": len(train_groups),
        "Test_Pots": len(test_groups),
        "Baseline_Balanced_Accuracy": baseline_score
    })


# ============================================================
# AGGREGATE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(importance_results)
fold_df = pd.DataFrame(fold_results)

summary = (
    importance_df
    .groupby("Feature")
    .agg(
        Mean_Importance=("Importance_Mean", "mean"),
        Std_Importance=("Importance_Mean", "std"),
        Positive_Folds=(
            "Importance_Mean",
            lambda values: int((values > 0).sum())
        )
    )
    .reset_index()
)

summary["Sensor"] = summary["Feature"].apply(
    lambda x: "VNIR" if x.startswith("VNIR_") else "SWIR"
)

def extract_wavelength(feature):
    match = re.search(r"_mean_([0-9.]+)nm$", feature)
    return float(match.group(1)) if match else np.nan

summary["Wavelength_nm"] = summary["Feature"].apply(
    extract_wavelength
)

summary = summary.sort_values(
    "Mean_Importance",
    ascending=False
).reset_index(drop=True)

# Save detailed results.
importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "per_fold_feature_importance.csv"),
    index=False
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance_summary.csv"),
    index=False
)

fold_df.to_csv(
    os.path.join(OUTPUT_DIR, "cross_validation_scores.csv"),
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 75)
print("TOP 25 SPECTRAL FEATURES")
print("=" * 75)

print(
    summary[
        [
            "Feature",
            "Sensor",
            "Wavelength_nm",
            "Mean_Importance",
            "Std_Importance",
            "Positive_Folds"
        ]
    ].head(25).to_string(index=False)
)


# ============================================================
# SENSOR-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SENSOR-LEVEL FEATURE IMPORTANCE")
print("=" * 75)

for sensor in ["VNIR", "SWIR"]:
    sensor_df = summary[summary["Sensor"] == sensor]

    print(f"\n{sensor}")
    print("-" * 50)

    print("Number of features:", len(sensor_df))

    print(
        sensor_df[
            [
                "Wavelength_nm",
                "Mean_Importance",
                "Positive_Folds"
            ]
        ].head(10).to_string(index=False)
    )


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

scores = fold_df["Baseline_Balanced_Accuracy"]

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)

print(f"Mean balanced accuracy: {scores.mean():.4f}")
print(f"Standard deviation:     {scores.std():.4f}")


# ============================================================
# PLOTS
# ============================================================

import matplotlib.pyplot as plt

top_features = summary.head(20).sort_values(
    "Mean_Importance",
    ascending=True
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["Feature"],
    top_features["Mean_Importance"],
    xerr=top_features["Std_Importance"].fillna(0)
)

plt.xlabel("Mean decrease in balanced accuracy")
plt.ylabel("Spectral feature")
plt.title("Top 20 Grouped-CV Permutation Importances")
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "top_20_permutation_importance.png"
)

plt.savefig(plot_path, dpi=300)
plt.close()

# Separate wavelength plots for each sensor.
for sensor in ["VNIR", "SWIR"]:

    sensor_df = summary[
        summary["Sensor"] == sensor
    ].sort_values("Wavelength_nm")

    plt.figure(figsize=(12, 5))

    plt.plot(
        sensor_df["Wavelength_nm"],
        sensor_df["Mean_Importance"]
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean decrease in balanced accuracy")
    plt.title(f"{sensor} Wavelength Permutation Importance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{sensor}_wavelength_importance.png"
        ),
        dpi=300
    )

    plt.close()


print("\n" + "=" * 75)
print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
print("=" * 75)

print("\nOutput directory:", os.path.abspath(OUTPUT_DIR))
print("1. feature_importance_summary.csv")
print("2. per_fold_feature_importance.csv")
print("3. cross_validation_scores.csv")
print("4. top_20_permutation_importance.png")
print("5. VNIR_wavelength_importance.png")
print("6. SWIR_wavelength_importance.png")

print("\nInterpretation caution:")
print(
    "Permutation importance measures predictive usefulness, "
    "not biological causation. Correlated spectral bands can share "
    "or dilute importance. Results are exploratory and based on "
    "the current single-date dataset."
)