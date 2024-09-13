from torchvision import models
import torch.nn as nn


class RESNET18InVit16Custom(nn.Module):
    def __init__(self, n_classes, hidden_dim):
        super(RESNET18InVit16Custom, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.model(x)
