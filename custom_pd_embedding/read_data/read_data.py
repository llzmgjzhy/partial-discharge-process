import os
from multimaps import MyHttp
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import json


def extract_datetime_from_filename(filename):
    """
    Extracts the datetime object from a given filename.
    Assumes the filename format is 'prefix_YYYYMMDDHHMMSSfff.dat'
    where 'prefix' can be any string and 'fff' are milliseconds.
    Returns the datetime as a string in the format 'YYYY-MM-DD HH:MM:SS.fff'.
    """
    try:
        # Split the filename and remove the file extension
        datetime_part = filename.split("_")[-1].split(".")[0]

        # Parse the datetime
        # Assuming the format is "YYYYMMDDHHMMSSfff"
        datetime_obj = datetime.strptime(datetime_part, "%Y%m%d%H%M%S%f")

        # Format the datetime as a string
        datetime_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")

        return datetime_str
    except (IndexError, ValueError):
        # Return None if the filename format is incorrect
        return None


def extract_info(file_path: str, label: int):
    filename = os.path.basename(file_path)
    filetime = extract_datetime_from_filename(filename)
    data_info = {
        "device_type": 4,
        "detection_type": "AA",
        "protocol_ver": "1.0",
        "file_name": filename,
        "file_time": filetime,
        "label": label,
    }
    return data_info


def check_single_info(path):
    single_path = path
    if os.path.exists(single_path):
        with open(single_path, "rb") as file:
            data = file.read()
            try:
                Ins = MyHttp()
                data_info = extract_info(single_path)
                print(data_info["file_name"])
                instrument_serial_number = Ins.check_single_info(data)
                print(instrument_serial_number)
            except Exception as e:
                ret_msg = {"error": f"数据读取失败: {e}", "status_code": 301}
                print(ret_msg)
    else:
        print("file did not exist")


def get_argparse():
    parser = ArgumentParser(description="Process data")
    parser.add_argument(
        "--path",
        type=str,
        help="The path to the data file",
    )
    return parser.parse_args()


DISCHARGE_TYPE = [
    "无局放的信号",
    "干扰",
    "固体绝缘局放",
    "尖端电晕",
    "金属颗粒局放",
    "悬浮电位局放",
]

DATA_PATH = r"E:\Graduate\projects\partial_discharge_monitoring_20230904\research\processing-paradigm\data\AI_data_train"


def main(choose_type: str = "all", map_type: str = "0x21"):
    """
    read AI lab data from specified dir path

    can specific discharge type and map_type"""
    root_dir = DATA_PATH

    dir_paths = []  # [path,label] in dir_paths
    # get path list from specified dir path
    for root, dirs, files in os.walk(root_dir):
        if choose_type == "all" or choose_type in root:
            discharge_label = [
                DISCHARGE_TYPE.index(discharge_type)
                for discharge_type in DISCHARGE_TYPE
                if discharge_type in root
            ]
            dir_paths.extend(
                [
                    [os.path.join(root, file), discharge_label[0]]
                    for file in files
                    if file.endswith(".dat")
                    and any(discharge_type in root for discharge_type in DISCHARGE_TYPE)
                ]
            )

    all_data = []
    # read data from path list
    for i, path in enumerate(dir_paths):
        print(path)
        file_path = path[0].strip()
        label = path[1]
        print(label)
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                data = file.read()
                try:
                    Ins = MyHttp()
                    data_info = extract_info(file_path, label)
                    Ins.process_complete_data(data, data_info)
                    all_data.append(Ins.data)
                except Exception as e:
                    ret_msg = {"error": f"数据读取失败: {e}", "status_code": 301}
                    print(ret_msg)

    with open("./storage/all_ai_data_train.json", "w") as f:
        json.dump(all_data, f)


if __name__ == "__main__":
    main()
    # with open("./storage/all_ai_data_train.json", "r") as f:
    #     data = json.load(f)
    # print(data[0])
