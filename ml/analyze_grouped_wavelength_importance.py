
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "outputs/ml_datasets/multisensor_leaf_ml_dataset.csv"
OUTPUT_DIR = "artifacts/grouped_wavelength_importance"

WINDOW_SIZE_NM = 25
N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("TERRASPECTRA - GROUPED WAVELENGTH IMPORTANCE")
print("=" * 75)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

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
    raise ValueError("Unknown or missing treatment labels.")

y = target.astype(int)
groups = df["Pot"].astype(str)

feature_columns = [
    col for col in df.columns
    if col.startswith("VNIR_mean_")
    or col.startswith("SWIR_mean_")
]

X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError("Missing values in spectral features.")


# ============================================================
# BUILD SPECTRAL WINDOWS
# ============================================================

def get_wavelength(feature):
    match = re.search(r"_mean_([0-9.]+)nm$", feature)

    if not match:
        raise ValueError(f"Cannot extract wavelength: {feature}")

    return float(match.group(1))


feature_info = []

for index, feature in enumerate(feature_columns):

    sensor = "VNIR" if feature.startswith("VNIR_") else "SWIR"
    wavelength = get_wavelength(feature)

    window_start = (
        np.floor(wavelength / WINDOW_SIZE_NM)
        * WINDOW_SIZE_NM
    )

    window_end = window_start + WINDOW_SIZE_NM

    feature_info.append({
        "Feature_Index": index,
        "Feature": feature,
        "Sensor": sensor,
        "Wavelength_nm": wavelength,
        "Window_Start_nm": window_start,
        "Window_End_nm": window_end
    })

feature_info = pd.DataFrame(feature_info)

window_groups = []

for (sensor, start, end), subset in feature_info.groupby(
    ["Sensor", "Window_Start_nm", "Window_End_nm"]
):

    window_groups.append({
        "Sensor": sensor,
        "Window_Start_nm": start,
        "Window_End_nm": end,
        "Feature_Indices": subset["Feature_Index"].tolist(),
        "Number_of_Bands": len(subset)
    })

print("\nDATASET INFORMATION")
print("-" * 50)
print("Records:", len(X))
print("Total spectral bands:", X.shape[1])
print("VNIR bands:", sum(c.startswith("VNIR_") for c in feature_columns))
print("SWIR bands:", sum(c.startswith("SWIR_") for c in feature_columns))
print("Unique pots:", groups.nunique())
print("Spectral window size:", WINDOW_SIZE_NM, "nm")
print("Number of spectral windows:", len(window_groups))


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    ))
])

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# GROUPED PERMUTATION
# ============================================================

results = []
fold_scores = []

print("\nGROUPED PERMUTATION CROSS-VALIDATION")
print("-" * 75)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_pots = set(groups.iloc[train_idx])
    test_pots = set(groups.iloc[test_idx])

    if train_pots.intersection(test_pots):
        raise RuntimeError("Pot leakage detected.")

    model.fit(X_train, y_train)

    baseline_predictions = model.predict(X_test)

    baseline_score = balanced_accuracy_score(
        y_test,
        baseline_predictions
    )

    print(
        f"\nFold {fold}/{N_SPLITS} | "
        f"Test records: {len(test_idx)} | "
        f"Baseline BA: {baseline_score:.4f}"
    )

    fold_scores.append({
        "Fold": fold,
        "Baseline_Balanced_Accuracy": baseline_score,
        "Train_Pots": len(train_pots),
        "Test_Pots": len(test_pots)
    })

    X_test_array = X_test.to_numpy()

    for group_number, window in enumerate(window_groups):

        indices = window["Feature_Indices"]

        importance_values = []

        for repeat in range(N_REPEATS):

            rng = np.random.default_rng(
                RANDOM_STATE
                + fold * 10000
                + group_number * 100
                + repeat
            )

            permutation = rng.permutation(len(X_test_array))

            permuted_array = X_test_array.copy()

            # Shuffle all bands in this spectral window
            # using the same row permutation.
            permuted_array[:, indices] = (
                X_test_array[permutation][:, indices]
            )

            permuted_df = pd.DataFrame(
                permuted_array,
                columns=feature_columns
            )

            predictions = model.predict(permuted_df)

            permuted_score = balanced_accuracy_score(
                y_test,
                predictions
            )

            importance_values.append(
                baseline_score - permuted_score
            )

        results.append({
            "Fold": fold,
            "Sensor": window["Sensor"],
            "Window_Start_nm": window["Window_Start_nm"],
            "Window_End_nm": window["Window_End_nm"],
            "Number_of_Bands": window["Number_of_Bands"],
            "Importance_Mean": np.mean(importance_values),
            "Importance_Std": np.std(
                importance_values,
                ddof=1
            )
        })


# ============================================================
# AGGREGATE RESULTS
# ============================================================

results_df = pd.DataFrame(results)
fold_df = pd.DataFrame(fold_scores)

summary = (
    results_df
    .groupby([
        "Sensor",
        "Window_Start_nm",
        "Window_End_nm",
        "Number_of_Bands"
    ])
    .agg(
        Mean_Importance=("Importance_Mean", "mean"),
        Std_Importance=("Importance_Mean", "std"),
        Positive_Folds=(
            "Importance_Mean",
            lambda x: int((x > 0).sum())
        )
    )
    .reset_index()
)

summary = summary.sort_values(
    "Mean_Importance",
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "per_fold_window_importance.csv"),
    index=False
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "grouped_wavelength_summary.csv"),
    index=False
)

fold_df.to_csv(
    os.path.join(OUTPUT_DIR, "cross_validation_scores.csv"),
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("TOP 20 SPECTRAL WINDOWS")
print("=" * 75)

print(
    summary.head(20).to_string(index=False)
)

print("\nSENSOR-WISE TOP WINDOWS")
print("-" * 75)

for sensor in ["VNIR", "SWIR"]:

    sensor_summary = summary[
        summary["Sensor"] == sensor
    ]

    print(f"\n{sensor}")

    print(
        sensor_summary.head(10).to_string(index=False)
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

scores = fold_df["Baseline_Balanced_Accuracy"]

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)

print(f"Mean balanced accuracy: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")


# ============================================================
# VISUALIZATION
# ============================================================

for sensor in ["VNIR", "SWIR"]:

    sensor_df = summary[
        summary["Sensor"] == sensor
    ].sort_values("Window_Start_nm")

    plt.figure(figsize=(13, 5))

    centers = (
        sensor_df["Window_Start_nm"]
        + sensor_df["Window_End_nm"]
    ) / 2

    plt.bar(
        centers,
        sensor_df["Mean_Importance"],
        width=WINDOW_SIZE_NM * 0.8
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean decrease in balanced accuracy")
    plt.title(
        f"{sensor} Grouped Wavelength Importance "
        f"({WINDOW_SIZE_NM} nm windows)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{sensor}_grouped_wavelength_importance.png"
        ),
        dpi=300
    )

    plt.close()


top = summary.head(20).sort_values(
    "Mean_Importance",
    ascending=True
)

labels = [
    f"{row.Sensor} {row.Window_Start_nm:.0f}-"
    f"{row.Window_End_nm:.0f} nm"
    for row in top.itertuples()
]

plt.figure(figsize=(11, 8))

plt.barh(
    labels,
    top["Mean_Importance"]
)

plt.xlabel("Mean decrease in balanced accuracy")
plt.ylabel("Spectral window")
plt.title("Top 20 Grouped Spectral Windows")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_20_spectral_windows.png"),
    dpi=300
)

plt.close()


print("\n" + "=" * 75)
print("GROUPED WAVELENGTH ANALYSIS COMPLETED")
print("=" * 75)

print("Output directory:", os.path.abspath(OUTPUT_DIR))
print("1. grouped_wavelength_summary.csv")
print("2. per_fold_window_importance.csv")
print("3. cross_validation_scores.csv")
print("4. VNIR_grouped_wavelength_importance.png")
print("5. SWIR_grouped_wavelength_importance.png")
print("6. top_20_spectral_windows.png")

print("\nInterpretation:")
print(
    "Positive importance means shuffling that spectral window "
    "reduced model performance. Negative importance means it "
    "did not improve performance in that evaluation. These are "
    "predictive associations, not biological proof."
)

