import json
import numpy as np
from custom_pd_embedding.util import *
import torch

TRANSFORMER_INPUT_DIM = 768


def read_data(path: str, trace_steps: int = 9):
    """
    Read the data from the given path and return the content and label array.
    Arguments:
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

    Arguments:
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


def process_data(content_array: np.ndarray):
    """
    process content_array.original array is numpy array with shape (trace_steps,h,w).that is invalid for the transformer model.

    Arguments:
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

    return output_array