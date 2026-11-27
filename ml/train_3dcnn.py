
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# ============================================================
# TERRASPECTRA - COMPACT VNIR 3D CNN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs/3dcnn/training"
OUTPUT_DIR = ROOT / "outputs/3dcnn/model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.0005
PATIENCE = 8

DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# 1. MODEL
# ------------------------------------------------------------

class Compact3DCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (N, 1, 216, 11, 11)
            nn.Conv3d(
                1, 8,
                kernel_size=(7, 3, 3),
                padding=(3, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(
                8, 16,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(
                16, 32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

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


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

def load_array(name):
    return np.load(DATA_DIR / name, allow_pickle=False)


def load_dataset(split):

    X = load_array(f"X_{split}.npy")
    y = load_array(f"y_{split}.npy")

    if X.ndim != 4 or X.shape[1:] != (216, 11, 11):
        raise ValueError(f"Unexpected {split} input shape: {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError(f"{split} data contains non-finite values.")

    # Add channel dimension:
    # (N, 216, 11, 11) -> (N, 1, 216, 11, 11)
    X = torch.from_numpy(
        np.asarray(X, dtype=np.float32)[:, None, :, :, :]
    )

    y = torch.from_numpy(
        np.asarray(y, dtype=np.int64)
    )

    return X, y


# ------------------------------------------------------------
# 3. TRAINING FUNCTION
# ------------------------------------------------------------

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)
            loss = criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)

            total_loss += loss.item() * len(labels)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(
                torch.argmax(logits, dim=1).cpu().numpy()
            )
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    metrics = {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob)
    }

    return metrics


def main():

    print("=" * 70)
    print("TERRASPECTRA - VNIR 3D CNN TRAINING")
    print("=" * 70)

    print("\nDevice:", DEVICE)
    print("PyTorch:", torch.__version__)

    # --------------------------------------------------------
    # LOAD TRAIN / VALIDATION / TEST
    # --------------------------------------------------------

    X_train, y_train = load_dataset("train")
    X_val, y_val = load_dataset("val")
    X_test, y_test = load_dataset("test")

    print("\nDATASET SHAPES")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # MODEL AND OPTIMIZER
    # --------------------------------------------------------

    model = Compact3DCNN().to(DEVICE)

    parameter_count = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", f"{parameter_count:,}")

    # Balanced class weights based on training labels only
    counts = np.bincount(y_train.numpy(), minlength=2)

    class_weights = len(y_train) / (
        2.0 * np.maximum(counts, 1)
    )

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    print("\nSTARTING TRAINING")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_metrics = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"]
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val BA: {val_metrics['balanced_accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Save best model based on validation loss
        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "Compact3DCNN",
            "input_shape": [1, 216, 11, 11],
            "best_epoch": best_epoch,
            "seed": SEED
        },
        OUTPUT_DIR / "best_vnir_3dcnn.pth"
    )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    print("\nBest epoch:", best_epoch)

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_metrics = evaluate(
        model, test_loader, criterion
    )

    print("\nTest metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            logits = model(inputs.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                torch.argmax(logits, dim=1).cpu().numpy()
            )
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Non-infected", "Infected"],
            zero_division=0
        )
    )

    # Save test predictions for later leaf/pot aggregation
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "infected_probability": y_prob
    }).to_csv(
        OUTPUT_DIR / "test_patch_predictions.csv",
        index=False
    )

    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("\nSaved model and evaluation outputs:")
    print(OUTPUT_DIR)

    print("\n3D CNN TRAINING FINISHED")


if __name__ == "__main__":
    main()
