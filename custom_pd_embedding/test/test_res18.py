from argparse import ArgumentParser
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
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
from custom_pd_embedding.model import RESNET18
from custom_pd_embedding.train.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_now = datetime.now().strftime("%m-%d-%y_%H-%M-%S")

# tensorBoard
tensorBoard_path = "runs_resnet18"
writer_train = SummaryWriter(f"./{tensorBoard_path}/{time_now}/train")
writer_test = SummaryWriter(f"./{tensorBoard_path}/{time_now}/test")


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
        default=8,
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
        / "custom_pd_embedding/model/mlp/mlp_09-12-24_17-17-45.pth",
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
    return parser.parse_args()


def test(args, model, trainLoader, testLoader):
    # loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in testLoader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(testLoader)
                epoch_val_loss += val_loss / len(testLoader)

        # tensorBoard
        writer_test.add_scalar("loss", epoch_val_loss, epoch)
        writer_test.add_scalar("acc", epoch_val_accuracy, epoch)

        print(f" val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")


if __name__ == "__main__":
    # argparse and save config to json
    args = getArgparse()
    console_save_args_to_json(
        args,
        PROJECT_ROOT,
        time_now,
        tb_path=tensorBoard_path,
        args_json_path="custom_pd_embedding/test",
    )

    # get data
    content_array, label_array = read_data(args.path, trace_steps=args.trace_steps)
    processed_content_array = process_data_for_resnet(content_array)

    # making dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed_content_array,
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
    model = RESNET18(n_classes=len(np.unique(label_array))).to(device)
    model.load_state_dict(torch.load(args.resnet_path))

    # test model
    test(args, model, trainLoader, testLoader)