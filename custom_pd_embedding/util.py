from custom_pd_embedding.feature_extract import (
    ManmadeExtractor,
    ManmadeExtractorTensor,
    NetworkExtractorResnetFineTune,
)
import numpy as np
import torch


def construct_manmade_features(input_array: np.ndarray) -> np.ndarray:
    """
    Construct the manmade features from the input array.
    """
    manmadeExtractor = ManmadeExtractor()

    pulse_phase_descriptor = manmadeExtractor.pulse_phase_descriptor(input_array)
    skewness_cycle = manmadeExtractor.skewness_cycle(input_array)
    kurtosis_cycle = manmadeExtractor.kurtosis_cycle(input_array)
    peak_num = manmadeExtractor.peak_num(input_array)

    manmade_array = np.array(
        [
            [
                pulse_phase_descriptor[i]["pulse_count"],
                pulse_phase_descriptor[i]["amplitude_max"],
                pulse_phase_descriptor[i]["amplitude_sum"],
                pulse_phase_descriptor[i]["amplitude_mean"],
            ]
            for i in range(len(pulse_phase_descriptor))
        ]
    )

    manmade_vector = manmade_array.flatten()
    manmade_vector = np.append(
        manmade_vector, [skewness_cycle, kurtosis_cycle, peak_num]
    )

    return manmade_vector


def construct_manmade_features_tensor(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Construct the manmade features from the input tensor.
    """
    manmadeExtractor = ManmadeExtractorTensor()

    pulse_phase_descriptor = manmadeExtractor.pulse_phase_descriptor(input_tensor)
    skewness_cycle = manmadeExtractor.skewness_cycle(input_tensor)
    kurtosis_cycle = manmadeExtractor.kurtosis_cycle(input_tensor)
    peak_num = manmadeExtractor.peak_num(input_tensor)

    manmade_tensor = torch.stack(
        [
            pulse_phase_descriptor[i][:4]  # Directly slice the first four elements
            for i in range(pulse_phase_descriptor.shape[0])
        ]
    )

    # Flatten the tensor and concatenate with additional features
    additional_features = torch.tensor([skewness_cycle, kurtosis_cycle, peak_num])
    manmade_vector = torch.cat((manmade_tensor.flatten(), additional_features))

    return manmade_vector


def construct_network_features(
    input_array: torch.Tensor, n_classes: int = 6, network_dims: int = 768
) -> torch.Tensor:
    """
    Construct the network features from the input array.
    """
    networkExtractor = NetworkExtractorResnetFineTune()
    network_vector = networkExtractor.resnet18_feature(
        input_array, n_classes, network_dims
    )
    return network_vector
