from custom_pd_embedding.model import *
import numpy as np
from torch import Tensor


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
