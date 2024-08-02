from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from custom_pd_embedding.util import *

writer = SummaryWriter("./runs")

TRANSFORMER_INPUT_DIM = 768


if __name__ == "__main__":
    # read and prepare for data

    # extract manmade features
    manmade_vector = construct_manmade_features()
    # Load the model and extract the network features
    network_vector = networkExtractor

    # connect manmade and network features,making it available for the transformer

    # access to transformer and train transformer

    # save the model param
