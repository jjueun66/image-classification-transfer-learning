import torch
import torch.nn as nn
from torch.optim import Adam

from src.dataset import get_cifar10_loaders
from src.models import SimpleCNN
from src.train import train_one_epoch, evaluate
from src.utils import History, ensure_dir, save_history_csv, plot_train_val_curves, plot_and_save_confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp1] Using device: {device}")

    results_dir = ensure_dir("results")
    exp_name = "exp1_base_no_aug"

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir="./data",
        batch_size=128,
        val_ratio=0.1,
        num_workers=4,
        use_augmentation=False,
    )

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    best_val_acc = -1.0
    best_state = None

    hist = History([], [], [], [], [])

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        hist.epochs.append(epoch + 1)
        hist.train_loss.append(tr_loss)
        hist.train_acc.append(tr_acc)
        hist.val_loss.append(va_loss)
        hist.val_acc.append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[Exp1][{epoch+1:02d}/{num_epochs}] Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

    # best 모델 로드
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[Exp1] Best Val Acc: {best_val_acc:.4f}")
    print(f"[Exp1] Test Acc: {test_acc:.4f}")

    # 저장
    # 1) 모델
    torch.save(model.state_dict(), results_dir / f"{exp_name}_best.pt")

    # 2) history csv
    save_history_csv(hist, results_dir / f"{exp_name}_history.csv")

    # 3) curves
    plot_train_val_curves(hist, results_dir / f"{exp_name}_curves.png", title=exp_name)

    # 4) confusion matrix (test set)
    class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    plot_and_save_confusion_matrix(
        model, test_loader, device, class_names,
        results_dir / f"{exp_name}_confusion.png",
        normalize="true",
        title=f"{exp_name} (normalized)",
    )


if __name__ == "__main__":
    main()
