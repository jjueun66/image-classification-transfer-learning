import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        # 입력: (B, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 4, 4)

        x = x.view(x.size(0), -1)             # (B, 128*4*4)
        x = self.dropout(F.relu(self.fc1(x))) # (B, 256)
        x = self.fc2(x)                       # (B, num_classes)
        return x
