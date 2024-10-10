from argparse import ArgumentParser
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys

# make sure the path of import adequate
sys.path.insert(0, str(PROJECT_ROOT))
from custom_pd_embedding.read_data.ai_data_train.dataset import AItrainDataset
import timm
from custom_pd_embedding.model import MLP, RESNET18, ViT, weightVit
import matplotlib.pyplot as plt
from custom_pd_embedding.train.util import *
from upsetplot import plot, from_indicators
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_now = datetime.now().strftime("%m-%d-%y_%H-%M-%S")


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
    # model params
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="The learning rate of the model",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="vit model dim",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="vit model depth",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="vit model heads",
    )
    parser.add_argument(
        "--mlp-dim",
        type=int,
        default=64,
        help="vit model mlp-dim",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="The learning rate decay rate",
    )
    # dataset params
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=1,
        help="The num of backtracking images",
    )
    parser.add_argument(
        "--weightVit-trace-steps",
        type=int,
        default=4,
        help="The num of backtracking images",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=10,
        help="random seed to generate the train and test dataset",
    )
    parser.add_argument(
        "--is-multi",
        action="store_true",
        help="Is multi weightVit",
    )
    parser.add_argument(
        "--mlp-path",
        type=str,
        default=PROJECT_ROOT
        / "custom_pd_embedding/model/mlp/mlp_10-08-24_20-03-43.pth",
        help="The mlp model path",
    )
    parser.add_argument(
        "--resnet-path",
        type=str,
        default=PROJECT_ROOT
        / "custom_pd_embedding/model/resnet/resnet18/resnet18_09-27-24_14-48-05.pth",
        help="The resnet model path",
    )
    parser.add_argument(
        "--vit-path",
        type=str,
        default=PROJECT_ROOT
        / "custom_pd_embedding/model/vit/vit_base/vit16_09-14-24_15-08-47.pth",
        help="The vit model path",
    )
    parser.add_argument(
        "--weightVit-path",
        type=str,
        default=PROJECT_ROOT
        / "custom_pd_embedding/model/pdvit/pdvit_4_10-10-24_22-09-42.pth",
        help="The weightVit model path",
    )
    return parser.parse_args()


def test(
    args, mlp_model, resnet_model, vit_model, weightVit_model, trainLoader, testLoader
):

    with torch.no_grad():
        mlp_model.eval()
        resnet_model.eval()
        vit_model.eval()
        weightVit_model.eval()

        mlp_global_correct = []
        resnet_global_correct = []
        vit_global_correct = []
        weightVit_global_correct = []
        bagging_global_correct = []
        for data, label in testLoader:
            data = data.to(device)
            label = label.to(device)
            data_mlp = data[:, :, :403]
            data_mlp = data_mlp.squeeze(1)
            data_resnet = data[:, :, 403:10403]
            data_resnet = data_resnet.reshape(
                data_resnet.shape[0], data_resnet.shape[1], 100, 100
            )
            data_vit = data[:, :, 10403:]

            val_mlp_output = mlp_model(data_mlp)

            val_resnet_output = resnet_model(data_resnet)

            val_vit_output = vit_model(data_vit)

            weightVit_output = weightVit_model.inference(data)

            # ensemble learning--bagging
            val_mlp_pre = val_mlp_output.argmax(dim=1)
            val_resnet_pre = val_resnet_output.argmax(dim=1)
            val_vit_pre = val_vit_output.argmax(dim=1)

            ensemble_pre = torch.stack(
                [val_mlp_pre, val_resnet_pre, val_vit_pre], dim=1
            )
            ensemble_final_pre = torch.mode(ensemble_pre, dim=1)
            ensemble_correct = (ensemble_final_pre.values == label).item()
            bagging_global_correct.append(ensemble_correct)

            mlp_correct = (val_mlp_output.argmax(dim=1) == label).item()
            mlp_global_correct.append(mlp_correct)
            resnet_correct = (val_resnet_output.argmax(dim=1) == label).item()
            resnet_global_correct.append(resnet_correct)
            vit_correct = (val_vit_output.argmax(dim=1) == label).item()
            vit_global_correct.append(vit_correct)
            weightVit_correct = (weightVit_output.argmax(dim=1) == label).item()
            weightVit_global_correct.append(weightVit_correct)

        upset_data = {
            "mlp": mlp_global_correct,
            "resnet": resnet_global_correct,
            "vit": vit_global_correct,
            "bagging": bagging_global_correct,
            "weightVit": weightVit_global_correct,
        }

        # draw upset
        df = pd.DataFrame(upset_data)
        plot(from_indicators(df), subset_size="count", show_counts="%d")

        plt.title("upsetPlot Diagram of Classifiers Based on Correct Predictions")
        plt.show()


if __name__ == "__main__":
    # argparse and save config to json
    args = getArgparse()

    # get data
    if os.path.exists(
        PROJECT_ROOT / f"data/storage/weightVit_{args.trace_steps}steps.json"
    ):
        content_array, label_array = read_processed_data_from_json(
            PROJECT_ROOT / f"data/storage/weightVit_{args.trace_steps}steps.json"
        )
    else:
        raise FileNotFoundError(
            "To speed up data processing,please save the data in advance"
        )

    # making dataset
    X_train, X_test, y_train, y_test = train_test_split(
        content_array,
        label_array,
        train_size=0.75,
        random_state=args.random_seed,
        stratify=label_array,
    )
    trainDataset = AItrainDataset(X_train, y_train)
    testDataset = AItrainDataset(X_test, y_test)

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    # model
    mlp_model = MLP(input_size=MLP_INPUT_DIM, output_size=6).to(device)
    mlp_model.load_state_dict(torch.load(args.mlp_path))

    resnet_model = RESNET18(n_classes=6).to(device)
    resnet_model.load_state_dict(torch.load(args.resnet_path))

    vit_model = ViT(
        image_size=16,
        patch_size=16,
        num_classes=6,
        dim=64,
        depth=4,
        heads=4,
        mlp_dim=64,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)
    vit_model.load_state_dict(torch.load(args.vit_path))

    weightVit_model = weightVit(
        num_classifiers=3,
        num_classes=6,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=0.1,
        emb_dropout=0.1,
        mlp_path=args.mlp_path,
        resnet_path=args.resnet_path,
        vit_path=args.vit_path,
        is_multi=args.is_multi,
        trace_steps=args.weightVit_trace_steps,
    ).to(device)
    weightVit_model.load_state_dict(torch.load(args.weightVit_path))

    # train transformer
    test(
        args,
        mlp_model,
        resnet_model,
        vit_model,
        weightVit_model,
        trainLoader,
        testLoader,
    )
