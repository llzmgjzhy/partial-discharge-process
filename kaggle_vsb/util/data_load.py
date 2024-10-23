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

# make sure the path of import adequate
sys.path.insert(0, str(PROJECT_ROOT))


def getArgparse():
    parser = ArgumentParser(description="load vsb data")
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
    parser.add_argument(
        "--save-path",
        type=str,
        default="three-phases",
        help="The path to the data file",
    )
    parser.add_argument(
        "--phase-num",
        type=int,
        default=3,
        help="The phase num for each signal",
    )
    return parser.parse_args()


def get_phase_data(df, meta_data, args):
    num_cols = df.shape[1]
    for i in range(0, num_cols, args.phase_num):
        for j in range(args.phase_num):
            if i + j < num_cols:
                plt.plot(
                    df.iloc[:, i + j],
                    label=f"phase {i + j}",
                    color=plt.cm.tab10(j),
                )
        plt.legend()
        plt.title(f"Signals {i} to {i+args.phase_num-1}")
        plt.title(f"Signals {i} to {i + args.phase_num-1}")
        label_path = "fault" if meta_data["target"][i] == 1 else "normal"
        plt.savefig(
            PROJECT_ROOT
            / "figs/vsb-power-line-origin"
            / f"{args.save_path}/train"
            / f"{label_path}/signals_{i}_to_{i + args.phase_num-1}.png"
        )
        plt.clf()


if __name__ == "__main__":

    args = getArgparse()

    # meta data
    meta_data_train = pd.read_csv(args.meta_path)

    df_train = pq.read_pandas(
        args.path,
    ).to_pandas()

    # data
    get_phase_data(df_train, meta_data_train, args)
