from argparse import ArgumentParser
import pyarrow.parquet as pq
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

# make sure the path of import adequate
sys.path.insert(0, str(PROJECT_ROOT))


def getArgparse():
    parser = ArgumentParser(
        description="Test the classification quality on the vsb dataset through kaggle data process procedure and mine vit model "
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


if __name__ == "__main__":
    args = getArgparse()

    # load data
    meta_data_train = pd.read_csv(args.meta_path)

    df_train = pq.read_pandas(
        args.path,
    ).to_pandas()
