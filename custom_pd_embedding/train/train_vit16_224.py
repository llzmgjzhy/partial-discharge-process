from argparse import ArgumentParser
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

# make sure the path of import include util file
sys.path.insert(0, str(PROJECT_ROOT))
from util import *


TRANSFORMER_INPUT_DIM = 768


def getArgparse():
    parser = ArgumentParser(description="Train vit")
    parser.add_argument(
        "--path",
        type=str,
        default=r"E:\Graduate\projects\partial_discharge_monitoring_20230904\research\processing-paradigm\data\storage\all_ai_data_train.json",
        help="The path to the data file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # read and prepare for data
    args = getArgparse()
    try:
        with open(args.path, "r") as f:
            pd_data = json.load(f)
    except Exception as e:
        print("catch data file error", e)

    # making dataset
    dataset = load_dataset(pd_data)
    # extract manmade features
    # manmade_vector = construct_manmade_features()
    # Load the model and extract the network features
    # network_vector = networkExtractor

    # connect manmade and network features,making it available for the transformer

    # access to transformer and train transformer

    # save the model param
