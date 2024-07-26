from model import *
import numpy as np


class NetworkExtractor:
    """
    Extract the network features from the input array.
    """

    def __init__(self) -> None:
        pass

    def resnet18_feature(input_array: np.ndarray, feature_num: int) -> int:
        """
        extract the features by the resnet18 model.

        result'dim is depend on the param feature_num,which represent the size of dim that the transformer can accept in addition to artificial features dim.
        """
        # check the input is whether valid
        if len(input_array.ndim) != 2:
            raise ValueError(
                "During resnet18 feature extract,Input array must have 2 dimensions"
            )

        # calculate the resnet18 feature of the input array
        resnet18_feature = 0
        resnet18_feature = RESNET18(n_classes=feature_num)

        return resnet18_feature
