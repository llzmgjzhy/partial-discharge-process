import os
from multimaps import MyHttp
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")


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


def extract_info(file_path):
    filename = os.path.basename(file_path)
    filetime = extract_datetime_from_filename(filename)
    data_info = {
        "device_type": 4,
        "detection_type": "AA",
        "protocol_ver": "1.0",
        "file_name": filename,
        "file_time": filetime,
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


def main():
    root_dir = r"E:\Graduate\projects\partial_discharge_monitoring_20230904\research\processing-paradigm\data\AI_data_train\固体绝缘局放\内部放电\强度中"
    dir_paths = []
    # get path list from specified dir path
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            dest_dir = os.path.join(root, file)
            dir_paths.append(dest_dir)
            break

    # read data from path list
    for i, path in enumerate(dir_paths):
        path = path.strip()
        if os.path.exists(path):
            with open(path, "rb") as file:
                data = file.read()
                try:
                    Ins = MyHttp()
                    data_info = extract_info(path)
                    Ins.process_complete_data(data, data_info)
                    img_example = np.array(Ins.data[1])
                    img_example = img_example[np.newaxis, :, :]
                    writer.add_image("img2", img_tensor=img_example,global_step=2)
                    writer.close()
                except Exception as e:
                    ret_msg = {"error": f"数据读取失败: {e}", "status_code": 301}
                    print(ret_msg)
 

if __name__ == "__main__":
    main()
    # for i in range(200):
    #     writer.add_scalar("data1", i*1.2+12, i)
