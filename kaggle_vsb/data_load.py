from argparse import ArgumentParser
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# make sure the path of import adequate
sys.path.insert(0, str(PROJECT_ROOT))


def getArgparse():
    parser = ArgumentParser(
        description="Train vit using ai train data by extract manmade and network features"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=PROJECT_ROOT / "data/vsb-power-line-fault-detection/train.parquet",
        help="The path to the data file",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=PROJECT_ROOT / "data/vsb-power-line-fault-detection/metadata_train.csv",
        help="The path to the data file",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = getArgparse()

    # meta data
    meta_data_train = pd.read_csv(args.meta_path)

    df_train = pq.read_pandas(
        args.path,
    ).to_pandas()

    # data
    num_cols = df_train.shape[1]
    for i in range(0, num_cols, 3):
        for j in range(3):
            if i + j < num_cols:
                plt.plot(
                    df_train.iloc[:, i + j],
                    label=f"phase {i + j}",
                    color=plt.cm.tab10(j),
                )
        plt.legend()
        plt.title(f"Signals {i} to {i+2}")
        if meta_data_train["target"][i] == 1:
            plt.savefig(
                PROJECT_ROOT
                / f"figs/vsb-power-line-origin/train/fault/signals_{i}_to_{i+2}.png"
            )
        else:
            plt.savefig(
                PROJECT_ROOT
                / f"figs/vsb-power-line-origin/train/normal/signals_{i}_to_{i+2}.png"
            )
        plt.clf()
