from argparse import ArgumentParser
import torch
from sklearn.model_selection import train_test_split
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("./runs")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys

# make sure the path of import adequate
sys.path.insert(0, str(PROJECT_ROOT))
from custom_pd_embedding.read_data.ai_data_train.dataset import AItrainDataset
from custom_pd_embedding.model import ViT
from custom_pd_embedding.train.util import *


def getArgparse():
    parser = ArgumentParser(
        description="Train vit using ai train data by extract manmade and network features"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=PROJECT_ROOT / "data/storage/all_ai_data_train.json",
        help="The path to the data file",
    )
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=9,
        help="The num of backtracking images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # read and prepare for data
    args = getArgparse()

    content_array, label_array = read_data(args.path, trace_steps=9)
    # process content_array
    processed_content_array = process_data(content_array)
    # for i in range(len(content_array)):
    #     for j in range(len(content_array[i])):
    #         # extract manmade features
    #         manmade_vector = construct_manmade_features(content_array[i][j])

    #         # extract the network features
    #         # confirm network dims
    #         network_feature_nums = TRANSFORMER_INPUT_DIM - len(manmade_vector)
    #         tensor_array = torch.from_numpy(content_array[i][j]).float()
    #         network_vector = construct_network_features(
    #             content_array[i][j], network_dims=network_feature_nums
    #         )
    #         print(network_vector)
    #         break

    # making dataset
    # random_state = 10
    # X_train, X_test, y_train, y_test = train_test_split(
    #     content_array,
    #     label_array,
    #     train_size=0.8,
    #     random_state=random_state,
    #     stratify=label_array,
    # )
    # trainDataset = AItrainDataset(X_train, y_train)
    # testDataset = AItrainDataset(X_test, y_test)
    # v = ViT(
    #     image_size=256,
    #     patch_size=32,
    #     num_classes=1000,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1,
    # )

    # concat manmade and network features,making it available for the transformer

    # access to transformer and train transformer

    # save the model param
