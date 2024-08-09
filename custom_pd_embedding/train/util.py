import json
import numpy as np
from custom_pd_embedding.util import *
import torch
import torchvision.transforms as transforms

TRANSFORMER_INPUT_DIM = 768
ARGS_JSON_PATH = "custom_pd_embedding/train"


def read_data(path: str, trace_steps: int = 9):
    """
    Read the data from the given path and return the content and label array.

    arguments:
    path: The path to the data file.

    trace_steps: Specify the number of backtracking images.If 9,indicating that 9 images of the same category will be input as the one image into the Vit model,with an input dimension of (b,9,768).768 is the original input dimension of the transformer model.

    return: content_array,label_array.dims of content_array is (trace_steps,h,w)

    """
    try:
        with open(path, "r") as f:
            read_data = json.load(f)

        # get origin data
        content_array = np.array([item[0] for item in read_data])
        label_array = np.array([item[1] for item in read_data])
        label_array = label_array.astype(np.int64)

        unique_labels = np.unique(label_array)
        response_content_array = []
        response_label_array = []
        for label in unique_labels:
            # get the indices of the label
            indices = np.where(label_array == label)[0]
            for i in range(0, len(indices), trace_steps):
                end_index = i + trace_steps
                # append data depending on the end_index
                if end_index <= len(indices):
                    response_content_array.append(content_array[indices[i:end_index]])
                else:
                    response_content_array.append(content_array[indices[-trace_steps:]])
                response_label_array.append(label)

        return np.array(response_content_array), np.array(response_label_array)
    except Exception as e:
        print("catch data file error", e)


def prepare_input_for_resnet(input_array: np.ndarray):
    """
    prepare for input array:regulation the input array and add the channel dim to match the input of resnet model.

    arguments:
    input_array: The input array to be prepared.
    """
    if input_array.ndim != 2:
        raise ValueError(
            "During resnet18 feature prepare,Input array must have 2 dimensions"
        )

    # Check if the denominator is zero
    if np.max(input_array) - np.min(input_array) == 0:
        normalized_array = np.zeros_like(input_array)
    else:
        normalized_array = (
            2
            * (
                (input_array - np.min(input_array))
                / (np.max(input_array) - np.min(input_array))
            )
            - 1
        )

    input_tensor = torch.from_numpy(normalized_array).unsqueeze(0).unsqueeze(0).float()

    return input_tensor


def process_data_for_resnet(content_array: np.ndarray):
    """
    process content_array.original array is numpy array with shape (trace_steps,h,w).that is invalid for the resnet model.

    arguments:
    content_array: The content array to be processed.

    procedure:
    regulation the input array,add the channel dim to match the input of resnet model and turn the type into float32.
    """
    count, trace_steps = content_array.shape[:2]
    output_array = np.zeros_like(content_array)

    for i in range(count):
        for j in range(trace_steps):
            network_input_array = prepare_input_for_resnet(content_array[i][j])
            output_array[i][j] = network_input_array

    return output_array.astype(np.float32)


def process_data_for_vit(content_array: np.ndarray):
    """
    process content_array.original array is numpy array with shape (trace_steps,h,w).that is invalid for the vit model.

    arguments:
    content_array: The content array to be processed.

    procedure:

    """
    tensor_array = torch.from_numpy(content_array)
    output_tensor = torch.zeros_like(tensor_array)

    # define resize transform
    resize_transform = transforms.Resize((224, 224))

    output_tensor = resize_transform(tensor_array)
    if output_tensor.size(1) == 1:
        output_tensor = output_tensor.repeat(1, 3, 1, 1)

    return output_tensor.to(dtype=torch.float32)


def process_data(content_array: np.ndarray):
    """
    process content_array.original array is numpy array with shape (trace_steps,h,w).that is invalid for the transformer model.

    arguments:
    content_array: The content array to be processed.

    procedure:
    reshape the content_array to (trace_steps,TRANSFORMER_INPUT_DIM)
    """

    # get the dims of content_array
    # count is content num,trace_steps is the num of backtracking images
    count, trace_steps = content_array.shape[:2]

    # create new numpy array,confirm the first two dim equal
    output_array = np.zeros((count, trace_steps, TRANSFORMER_INPUT_DIM))

    for i in range(count):
        for j in range(trace_steps):
            # extract manmade features
            manmade_vector = construct_manmade_features(content_array[i][j])

            # extract the network features
            # confirm network dims
            network_feature_nums = TRANSFORMER_INPUT_DIM - len(manmade_vector)
            # regulation the input array to the resnet model
            network_input_array = prepare_input_for_resnet(
                np.array(content_array[i][j])
            )
            network_vector = construct_network_features(
                network_input_array, network_dims=network_feature_nums
            )

            if network_vector.is_cuda:
                network_vector = network_vector.cpu()

            network_vector = network_vector.detach().numpy()
            # concat manmade and network features
            concat_vector = np.append(manmade_vector, network_vector)
            # update the content_array using concat vector
            output_array[i][j] = concat_vector

    return output_array.astype(np.float32)


def save_processed_data_to_json(
    content_array: np.ndarray, label_array: np.ndarray, path: str
):
    """
    Save the processed data to the given path.

    arguments:
    content_array: The content array to be saved.
    label_array: The label array to be saved.
    path: The path to save the processed data.
    """
    concent_list = content_array.tolist()
    label_list = label_array.tolist()

    save_json = {"content": concent_list, "label": label_list}

    with open(path, "w") as f:
        json.dump(save_json, f)


def read_processed_data_from_json(path: str):
    """
    Read the processed data from the given path.

    """
    try:
        with open(path, "r") as f:
            read_data = json.load(f)
    except Exception as e:
        print("catch processed data json file error", e)

    processed_array = np.array(read_data["content"], dtype=np.float32)
    label_array = np.array(read_data["label"], dtype=np.int64)

    return processed_array, label_array


def console_save_args_to_json(args, root_path, time_now, tb_path: str = "runs"):
    """
    Save the args to the json file.

    arguments:
    args: The args to be saved.
    root_path: The project root path.
    time_now: The time now.
    """
    print("[Training params]")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    args_json = json.dumps({k: str(v) for k, v in args.__dict__.items()})

    # save the config to json
    with open(
        root_path / ARGS_JSON_PATH / tb_path / time_now / f"{time_now}.json", "w"
    ) as f:
        f.write(args_json)
