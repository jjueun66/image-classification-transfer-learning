from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


@dataclass
class History:
    epochs: List[int]
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history_csv(history: History, out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    lines = ["epoch,train_loss,train_acc,val_loss,val_acc\n"]
    for i in range(len(history.epochs)):
        lines.append(
            f"{history.epochs[i]},{history.train_loss[i]:.6f},{history.train_acc[i]:.6f},"
            f"{history.val_loss[i]:.6f},{history.val_acc[i]:.6f}\n"
        )
    out_csv.write_text("".join(lines), encoding="utf-8")


def plot_train_val_curves(history: History, out_png: str | Path, title: str = "") -> None:
    out_png = Path(out_png)

    # Accuracy curve
    plt.figure()
    plt.plot(history.epochs, history.train_acc, label="train_acc")
    plt.plot(history.epochs, history.val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title if title else "Train/Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_acc.png"))
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(history.epochs, history.train_loss, label="train_loss")
    plt.plot(history.epochs, history.val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title if title else "Train/Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.with_name(out_png.stem + "_loss.png"))
    plt.close()


@torch.no_grad()
def get_preds_and_labels(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds_all = []
    labels_all = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        preds_all.append(preds.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    return np.concatenate(preds_all), np.concatenate(labels_all)


def plot_and_save_confusion_matrix(
    model,
    loader,
    device,
    class_names: List[str],
    out_png: str | Path,
    normalize: str | None = None,  # None, "true", "pred", "all"
    title: str = "Confusion Matrix",
) -> None:
    out_png = Path(out_png)

    y_pred, y_true = get_preds_and_labels(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
