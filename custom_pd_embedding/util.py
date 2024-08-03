from custom_pd_embedding.feature_extract import ManmadeExtractor, NetworkExtractor
import numpy as np

manmadeExtractor = ManmadeExtractor()
networkExtractor = NetworkExtractor()


def construct_manmade_features(input_array: np.ndarray) -> np.ndarray:
    """
    Construct the manmade features from the input array.
    """
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


def constuct_network_features(
    input_array: np.ndarray, network_dims: int = 768
) -> np.ndarray:
    """
    Construct the network features from the input array.
    """
    network_vector = networkExtractor.resnet18_feature(input_array, network_dims)
    return network_vector
