from custom_pd_embedding.model import *
from torch import Tensor
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NetworkExtractor:
    """
    Extract the network features from the input array.
    """

    def __init__(self) -> None:
        pass

    def resnet18_feature(self, input_array: Tensor, feature_num: int) -> int:
        """
        extract the features by the resnet18 model.

        result'dim is depend on the param feature_num,which represent the size of dim that the transformer can accept in addition to artificial features dim.
        """

        # calculate the resnet18 feature of the input array
        resnet18 = RESNET18(n_classes=feature_num)
        network_vector = resnet18(input_array)

        return network_vector


class NetworkExtractorTensor:
    """
    Extract the network features from the input array.
    """

    def __init__(self) -> None:
        pass

    def resnet18_feature(self, input_tensor: Tensor, feature_num: int) -> Tensor:
        """
        extract the features by the resnet18 model.

        result'dim is depend on the param feature_num,which represent the size of dim that the transformer can accept in addition to artificial features dim.
        """

        # calculate the resnet18 feature of the input array
        resnet18 = RESNET18(n_classes=feature_num)
        network_vector = resnet18(input_tensor)

        return network_vector


class NetworkExtractorResnetFineTune:
    """
    Extract the network features from the input array.
    """

    def __init__(self) -> None:
        pass

    def resnet18_feature(
        self, input_array: Tensor, n_classes: int, feature_num: int
    ) -> Tensor:
        """
        extract the features by the resnet18 model.

        result'dim is depend on the param feature_num,which represent the size of dim that the transformer can accept in addition to artificial features dim.
        """

        # calculate the resnet18 feature of the input array
        resnet18 = RESNET18InVit16Custom(n_classes=n_classes, hidden_dim=feature_num)
        resnet18_path = (
            PROJECT_ROOT
            / "model/resnet/resnet18/resnet18InVitCustom_09-13-24_22-30-45.pth"
        )
        resnet18.load_state_dict(torch.load(resnet18_path))
        # move the model to the device
        device = input_array.device
        resnet18.to(device)
        # origin model architecture is fc relu fc,but now only need the first fc layer to get network features instead of classification
        resnet18.model.fc = nn.Sequential(resnet18.model.fc[0])
        network_vector = resnet18(input_array)

        return network_vector
