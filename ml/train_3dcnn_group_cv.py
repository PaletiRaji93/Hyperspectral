
from pathlib import Path
import json
import random
import copy
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PATCH_FILE = ROOT / "outputs/3dcnn/vnir_patches.npy"
INDEX_FILE = ROOT / "outputs/3dcnn/vnir_patch_index.csv"

OUTPUT_DIR = ROOT / "outputs/3dcnn/group_cv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5

BATCH_SIZE = 4
MAX_EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 0.0005

DEVICE = torch.device("cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)
torch.set_num_threads(2)

print("=" * 65)
print("VNIR 3D-CNN: 5-FOLD POT-GROUPED CROSS-VALIDATION")
print("=" * 65)
print("Device:", DEVICE)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading patch data...")

X_raw = np.load(PATCH_FILE, mmap_mode="r")
metadata = pd.read_csv(INDEX_FILE)

if len(X_raw) != len(metadata):
    raise ValueError(
        "Patch array and metadata row counts do not match."
    )

required_columns = [
    "Pot",
    "Disease_Label",
    "Sample_ID",
    "Leaf"
]

missing = [
    column for column in required_columns
    if column not in metadata.columns
]

if missing:
    raise ValueError(
        f"Missing metadata columns: {missing}"
    )

# Disease_Label is already binary:
# 0 = non-infected
# 1 = infected

y = metadata["Disease_Label"].astype(np.int64).to_numpy()

if not set(np.unique(y)).issubset({0, 1}):
    raise ValueError(
        "Disease_Label must contain only 0 and 1."
    )

groups = metadata["Pot"].astype(str).to_numpy()

print("Patch shape:", X_raw.shape)
print("Total patches:", len(y))
print("Unique pots:", len(np.unique(groups)))

print(
    "Patch class counts:",
    dict(zip(*np.unique(y, return_counts=True)))
)


# ============================================================
# 4. VERIFY POT LABEL CONSISTENCY
# ============================================================

pot_table = metadata[["Pot"]].copy()
pot_table["Disease_Label"] = y

label_counts = (
    pot_table.groupby("Pot")["Disease_Label"].nunique()
)

if (label_counts > 1).any():
    raise ValueError(
        "At least one pot contains multiple disease labels."
    )

pot_summary = (
    pot_table.groupby("Pot")
    .agg(Disease_Label=("Disease_Label", "first"))
    .reset_index()
)

print("\nPot-level class counts:")
print(
    pot_summary["Disease_Label"]
    .value_counts()
    .sort_index()
)

if len(pot_summary) < N_SPLITS:
    raise ValueError("Not enough pots for 5-fold CV.")

if pot_summary["Disease_Label"].value_counts().min() < N_SPLITS:
    raise ValueError(
        "Each class needs at least 5 pots for 5-fold stratified CV."
    )


# ============================================================
# 5. DATASET
# ============================================================

class HyperspectralDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.from_numpy(
            np.asarray(X, dtype=np.float32)
        ).unsqueeze(1)

        self.y = torch.from_numpy(
            np.asarray(y, dtype=np.int64)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 6. 3D-CNN MODEL
# ============================================================

class VNIR3DCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1, 8,
                kernel_size=(7, 3, 3),
                padding=(3, 1, 1)
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(
                8, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 7. FOLD-SPECIFIC NORMALIZATION
# ============================================================

def normalize_fold(
    X_train_raw,
    X_val_raw,
    X_test_raw
):

    train = np.asarray(
        X_train_raw,
        dtype=np.float32
    )

    mean = train.mean(
        axis=(0, 2, 3),
        keepdims=True
    )

    std = train.std(
        axis=(0, 2, 3),
        keepdims=True
    )

    std[std < 1e-6] = 1.0

    X_train = (train - mean) / std

    X_val = (
        np.asarray(X_val_raw, dtype=np.float32) - mean
    ) / std

    X_test = (
        np.asarray(X_test_raw, dtype=np.float32) - mean
    ) / std

    return X_train, X_val, X_test, mean, std


# ============================================================
# 8. TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    fold_number
):

    set_seed(SEED + fold_number)

    model = VNIR3DCNN().to(DEVICE)

    class_counts = np.bincount(
        y_train,
        minlength=2
    )

    if np.any(class_counts == 0):
        raise ValueError(
            "Training split is missing a class."
        )

    weights = len(y_train) / (
        2.0 * class_counts.astype(np.float32)
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    train_dataset = HyperspectralDataset(
        X_train,
        y_train
    )

    val_dataset = HyperspectralDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = []

    print(f"\nFold {fold_number}: training started")

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        train_losses = []

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        scheduler.step(val_loss)

        history.append({
            "fold": fold_number,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"Fold {fold_number} | "
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint was created."
        )

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


# ============================================================
# 9. GROUPED 5-FOLD CROSS-VALIDATION
# ============================================================

pot_groups = pot_summary["Pot"].astype(str).to_numpy()

pot_labels = (
    pot_summary["Disease_Label"].to_numpy()
)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

all_predictions = []
all_histories = []
fold_metrics = []

for fold, (train_pot_idx, test_pot_idx) in enumerate(
    skf.split(pot_groups, pot_labels),
    start=1
):

    train_pots = pot_groups[train_pot_idx]
    test_pots = pot_groups[test_pot_idx]

    train_pot_labels = pot_labels[train_pot_idx]

    train_pots_inner, val_pots = train_test_split(
        train_pots,
        test_size=0.20,
        random_state=SEED + fold,
        stratify=train_pot_labels
    )

    train_mask = np.isin(groups, train_pots_inner)
    val_mask = np.isin(groups, val_pots)
    test_mask = np.isin(groups, test_pots)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Verify no pot leakage.
    assert not set(groups[train_idx]) & set(groups[val_idx])
    assert not set(groups[train_idx]) & set(groups[test_idx])
    assert not set(groups[val_idx]) & set(groups[test_idx])

    print("\n" + "=" * 65)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 65)

    print("Training pots:", len(train_pots_inner))
    print("Validation pots:", len(val_pots))
    print("Testing pots:", len(test_pots))

    print(
        "Training patches:", len(train_idx),
        "| Validation patches:", len(val_idx),
        "| Testing patches:", len(test_idx)
    )

    X_train, X_val, X_test, mean, std = normalize_fold(
        X_raw[train_idx],
        X_raw[val_idx],
        X_raw[test_idx]
    )

    np.savez_compressed(
        OUTPUT_DIR / f"fold_{fold}_normalization.npz",
        mean=mean,
        std=std
    )

    model, history = train_one_fold(
        X_train,
        y[train_idx],
        X_val,
        y[val_idx],
        fold
    )

    all_histories.append(history)

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / f"fold_{fold}_model.pth"
    )

    # Predict held-out test patches.
    model.eval()

    test_dataset = HyperspectralDataset(
        X_test,
        y[test_idx]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    probabilities = []

    with torch.no_grad():

        for xb, _ in test_loader:

            xb = xb.to(DEVICE)

            logits = model(xb)

            probs = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    fold_df = metadata.iloc[test_idx].copy()

    fold_df["y_true"] = y[test_idx]
    fold_df["infected_probability"] = probabilities
    fold_df["y_pred"] = predictions
    fold_df["fold"] = fold

    all_predictions.append(fold_df)

    fold_ba = balanced_accuracy_score(
        y[test_idx],
        predictions
    )

    fold_f1 = f1_score(
        y[test_idx],
        predictions,
        zero_division=0
    )

    fold_auc = roc_auc_score(
        y[test_idx],
        probabilities
    )

    fold_metrics.append({

        "fold": fold,
        "train_pots": len(train_pots_inner),
        "validation_pots": len(val_pots),
        "test_pots": len(test_pots),
        "test_patches": len(test_idx),

        "accuracy": accuracy_score(
            y[test_idx],
            predictions
        ),

        "balanced_accuracy": fold_ba,
        "f1": fold_f1,
        "roc_auc": fold_auc
    })

    print(
        f"\nFold {fold} results: "
        f"BA={fold_ba:.4f}, "
        f"F1={fold_f1:.4f}, "
        f"AUC={fold_auc:.4f}"
    )

    # Release fold data and model from memory.
    del X_train, X_val, X_test
    del test_dataset, test_loader
    del model

    gc.collect()


# ============================================================
# 10. SAVE PATCH-LEVEL OUT-OF-FOLD PREDICTIONS
# ============================================================

oof = pd.concat(
    all_predictions,
    ignore_index=True
)

oof.to_csv(
    OUTPUT_DIR / "oof_patch_predictions.csv",
    index=False
)

pd.concat(
    all_histories,
    ignore_index=True
).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

fold_metrics_df = pd.DataFrame(fold_metrics)

fold_metrics_df.to_csv(
    OUTPUT_DIR / "fold_metrics.csv",
    index=False
)


# ============================================================
# 11. LEAF-LEVEL AND POT-LEVEL AGGREGATION
# ============================================================

leaf_predictions = (
    oof.groupby(
        ["Pot", "Sample_ID", "Leaf"],
        as_index=False
    )
    .agg(
        y_true=("y_true", "first"),
        infected_probability=(
            "infected_probability",
            "mean"
        )
    )
)

leaf_predictions["y_pred"] = (
    leaf_predictions["infected_probability"] >= 0.5
).astype(int)

pot_predictions = (
    leaf_predictions.groupby(
        "Pot",
        as_index=False
    )
    .agg(
        y_true=("y_true", "first"),
        infected_probability=(
            "infected_probability",
            "mean"
        )
    )
)

pot_predictions["y_pred"] = (
    pot_predictions["infected_probability"] >= 0.5
).astype(int)

leaf_predictions.to_csv(
    OUTPUT_DIR / "oof_leaf_predictions.csv",
    index=False
)

pot_predictions.to_csv(
    OUTPUT_DIR / "oof_pot_predictions.csv",
    index=False
)


# ============================================================
# 12. FINAL METRICS
# ============================================================

def calculate_metrics(
    name,
    y_true,
    y_pred,
    probabilities
):

    return {

        "level": name,
        "observations": int(len(y_true)),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred
        ).tolist()
    }


patch_metrics = calculate_metrics(
    "patch",
    oof["y_true"],
    oof["y_pred"],
    oof["infected_probability"]
)

leaf_metrics = calculate_metrics(
    "leaf",
    leaf_predictions["y_true"],
    leaf_predictions["y_pred"],
    leaf_predictions["infected_probability"]
)

pot_metrics = calculate_metrics(
    "pot",
    pot_predictions["y_true"],
    pot_predictions["y_pred"],
    pot_predictions["infected_probability"]
)

final_metrics = {
    "patch": patch_metrics,
    "leaf": leaf_metrics,
    "pot": pot_metrics
}

with open(
    OUTPUT_DIR / "overall_metrics.json",
    "w"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )


# ============================================================
# 13. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("5-FOLD GROUPED CROSS-VALIDATION COMPLETE")
print("=" * 65)

print("\nFold-wise results:")

print(
    fold_metrics_df[
        [
            "fold",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "roc_auc"
        ]
    ].round(4).to_string(index=False)
)

print("\nMean and standard deviation across folds:")

for metric in [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc"
]:

    print(
        f"{metric}: "
        f"{fold_metrics_df[metric].mean():.4f} "
        f"+/- "
        f"{fold_metrics_df[metric].std():.4f}"
    )

print("\nPooled out-of-fold results:")

for result in [
    patch_metrics,
    leaf_metrics,
    pot_metrics
]:

    print(f"\n{result['level'].upper()} LEVEL")
    print("Observations:", result["observations"])
    print("Accuracy:", round(result["accuracy"], 4))
    print(
        "Balanced accuracy:",
        round(result["balanced_accuracy"], 4)
    )
    print("F1:", round(result["f1"], 4))
    print("ROC-AUC:", round(result["roc_auc"], 4))
    print(
        "Confusion matrix:",
        result["confusion_matrix"]
    )

print("\nSaved results to:")
print(OUTPUT_DIR)

print("\nExperiment finished successfully.")
