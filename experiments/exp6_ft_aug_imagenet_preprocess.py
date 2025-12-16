import torch
import torch.nn as nn
from torch.optim import Adam

from src.dataset import get_cifar10_loaders_for_pretrained
from src.models import get_resnet18_finetune
from src.train import train_one_epoch, evaluate
from src.utils import (
    History,
    ensure_dir,
    save_history_csv,
    plot_train_val_curves,
    plot_and_save_confusion_matrix,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp6] Using device: {device}")

    results_dir = ensure_dir("results")
    exp_name = "exp6_ft_aug_imagenet_preprocess"

    # ResNet용 DataLoader (augmentation=True)
    train_loader, val_loader, test_loader = get_cifar10_loaders_for_pretrained(
        data_dir="./data",
        batch_size=128,
        val_ratio=0.1,
        num_workers=4,
        use_augmentation=True,
    )

    # ResNet18 partial fine-tuning (layer4 + fc)
    model = get_resnet18_finetune(num_classes=10, train_all=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

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

        print(f"[Exp6][{epoch+1:02d}/{num_epochs}] Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[Exp6] Best Val Acc: {best_val_acc:.4f}")
    print(f"[Exp6] Test Acc: {test_acc:.4f}")

    torch.save(model.state_dict(), results_dir / f"{exp_name}_best.pt")
    save_history_csv(hist, results_dir / f"{exp_name}_history.csv")
    plot_train_val_curves(hist, results_dir / f"{exp_name}_curves.png", title=exp_name)

    class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    plot_and_save_confusion_matrix(
        model, test_loader, device, class_names,
        results_dir / f"{exp_name}_confusion.png",
        normalize="true",
        title=f"{exp_name} (normalized)",
    )


if __name__ == "__main__":
    main()
