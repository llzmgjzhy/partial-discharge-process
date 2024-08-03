from argparse import ArgumentParser
import torch
from sklearn.model_selection import train_test_split
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import json
import numpy as np
from datasets import load_metric
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("./runs")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys

# make sure the path of import adequate
sys.path.insert(0, str(PROJECT_ROOT))
from custom_pd_embedding.util import *
from custom_pd_embedding.read_data.ai_data_train.dataset import AItrainDataset
from vit_pytorch import ViT


TRANSFORMER_INPUT_DIM = 768


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
    return parser.parse_args()


def dataProcessor():
    return None


def collate_fn(batch):
    return {
        "pixel_values": [x[0] for x in batch],
        "labels": [x[1] for x in batch],
    }


metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


if __name__ == "__main__":
    # read and prepare for data

    args = getArgparse()

    try:
        with open(args.path, "r") as f:
            read_data = json.load(f)
            content_array = np.array([iter[0] for iter in read_data])
            label_array = np.array([iter[1] for iter in read_data])
    except Exception as e:
        print("catch data file error", e)

    # making dataset
    dataset = AItrainDataset(content_array, label_array)
    random_state = 10
    X_train, X_test, y_train, y_test = train_test_split(
        content_array, label_array, random_state=random_state, stratify=label_array
    )
    trainDataset = AItrainDataset(X_train, y_train)
    testDataset = AItrainDataset(X_test, y_test)
    v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
    # extract manmade features
    # manmade_vector = construct_manmade_features()

    # Load the model and extract the network features
    # network_vector = networkExtractor

    # concat manmade and network features,making it available for the transformer

    # access to transformer and train transformer
    transformer_name_or_path = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(
        transformer_name_or_path,
        num_labels=len(label_array),
    )
    training_args = TrainingArguments(
        output_dir=PROJECT_ROOT / "custom_pd_embedding/model/save/vit16_224/",
        per_device_train_batch_size=16,
        eval_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=trainDataset,
        eval_dataset=testDataset,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

# save the model param
