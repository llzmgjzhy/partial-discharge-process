from torch.utils.tensorboard import SummaryWriter
from feature_extract import ManmadeExtractor, NetworkExtractor
import numpy as np
import argparse
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer


writer = SummaryWriter("./runs")

TRANSFORMER_INPUT_DIM = 768

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


if __name__ == "__main__":
    # read and prepare for data

    # extract manmade features
    manmade_vector = construct_manmade_features()
    # Load the model and extract the network features
    network_vector = networkExtractor

    # connect manmade and network features,making it available for the transformer

    # access to transformer and train transformer

    # save the model param
