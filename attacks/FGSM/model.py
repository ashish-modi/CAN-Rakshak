import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, stride=1, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=32, out_channels=32, stride=1, kernel_size=3, padding='valid'),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding='valid'),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        )

    def forward(self, x):
        return self.stem(x)


class InceptionResNetABlock(nn.Module):
    def __init__(self, in_channels=128, scale=0.17):
        super(InceptionResNetABlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.conv_up = nn.Conv2d(96, 128, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim=1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)


class ReductionA(nn.Module):
    def __init__(self, in_channels=128):
        super(ReductionA, self).__init__()
        self.branch0 = nn.Conv2d(in_channels=in_channels, out_channels=192, kernel_size=3, stride=2, padding='valid')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding='valid')
        )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch0, branch1, branch2], dim=1)


class InceptionResNetBBlock(nn.Module):
    def __init__(self, in_channels=448, scale=0.10):
        super(InceptionResNetBBlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding='same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding='same'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding='same')
        )
        self.conv_up = nn.Conv2d(in_channels=128, out_channels=448, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        mixed = torch.cat([branch0, branch1], dim=1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels=448, out_channels=128, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding='valid')
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=448, out_channels=128, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='valid')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=448, out_channels=128, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='valid')
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat([branch0, branch1, branch2, branch3], dim=1)


class InceptionResNetV1(nn.Module):
    """Surrogate/target IDS architecture used by the New_FGSM attack."""

    def __init__(self, num_classes=2):
        super(InceptionResNetV1, self).__init__()
        self.stem = InceptionStem()
        self.a_block = InceptionResNetABlock()
        self.b_block = InceptionResNetBBlock()
        self.red_a = ReductionA()
        self.red_b = ReductionB()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(896, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.a_block(x)
        x = self.red_a(x)
        x = self.b_block(x)
        x = self.red_b(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
