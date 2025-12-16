# src/models.py

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleCNN(nn.Module):
    """
    CIFAR-10용 간단한 Baseline CNN 모델 (from scratch).
    입력: (B, 3, 32, 32)
    출력: (B, num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 특징 추출부
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 16x16 -> 16x16
        self.pool  = nn.MaxPool2d(2, 2)  # 크기 절반

        # 완전연결층
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

        # NOTE: CIFAR-10 기준으로 설계 (입력 3x32x32)

    def forward(self, x):
        # 입력: (B, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 4, 4)

        x = x.view(x.size(0), -1)             # (B, 128*4*4)
        x = self.dropout(F.relu(self.fc1(x))) # (B, 256)
        x = self.fc2(x)                       # (B, num_classes)
        return x



def get_resnet18_finetune(num_classes: int = 10, train_all: bool = False):
    """
    ResNet-18 기반 Fine-tuning 모델.
    - train_all = False: layer4 + fc만 학습, 나머지 freeze
    - train_all = True : 전체 레이어 fine-tuning

    실험 3, 4에서는 일단 train_all=False (부분 fine-tuning)로 사용하는 걸 추천.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if not train_all:
        # 기본은 전체 freeze
        for name, param in model.named_parameters():
            param.requires_grad = False

            # layer4 블록과 fc만 학습
            if name.startswith("layer4") or name.startswith("fc"):
                param.requires_grad = True
    else:
        # 전체 fine-tune
        for param in model.parameters():
            param.requires_grad = True

    # FC 레이어는 항상 현재 데이터셋 클래스 수에 맞게 교체
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
