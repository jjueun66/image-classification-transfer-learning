import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    use_augmentation: bool = True,
):
    # CIFAR-10 평균/표준편차
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # train 데이터셋 (다운로드 자동)
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # test 데이터셋
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # train/val 나누기
    n_total = len(full_train_dataset)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # DataLoader 만들기
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_cifar10_loaders_for_pretrained(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    use_augmentation: bool = True,
):
    """
    ImageNet pretrained 모델(ResNet 등)에 맞춘 CIFAR-10 DataLoader.
    - 입력을 224x224로 맞춤
    - ImageNet mean/std로 normalize
    - train에만 augmentation 적용
    - val/test는 항상 deterministic transform(center crop) 적용
    """

    # ImageNet 평균/표준편차 (torchvision pretrained 기준)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 핵심: 같은 train split을 "transform만 다르게" 두 개로 만든다
    full_train_for_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    full_train_for_val = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,          # 이미 위에서 받았으니 보통 False로 둠
        transform=eval_transform, # val은 항상 eval transform
    )

    # test 데이터셋 (eval transform)
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    # train/val split 인덱스 생성 (seed 고정)
    n_total = len(full_train_for_train)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    # 같은 인덱스를 서로 다른 transform dataset에 적용
    train_dataset = torch.utils.data.Subset(full_train_for_train, train_idx)
    val_dataset   = torch.utils.data.Subset(full_train_for_val,   val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


