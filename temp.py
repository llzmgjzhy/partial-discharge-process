import numpy as np
from torchvision import models

model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
print(in_features)