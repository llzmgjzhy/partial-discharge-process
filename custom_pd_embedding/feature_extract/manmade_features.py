# default input is an numpy array: d[m][n]
# m is the phase window,n is the quantify amplitude
# all function running procedures is based on the input

import numpy as np
import torch


class ManmadeExtractor:
    """
    Extract the manmade features from the input array.
    """

    def __init__(self) -> None:
        pass

    def pulse_phase_descriptor(self, input_array: np.ndarray) -> list:
        """
        Extract the phase descriptor of pulses of dimensions specified in the input array.

        include pulse_sum ,pulse_amplitude_max,pulse_amplitude_sum,pulse_amplitude_mean
        """

        # check the input is whether valid
        if input_array.ndim != 2:
            raise ValueError(
                "During pulse phase window feature extract,Input array must have 2 dimensions"
            )

        basic_descriptor = []
        for i in range(input_array.shape[0]):
            # calculate the pulse count
            pulse_count = np.sum(input_array[i])

            # calculate the pulse amplitude max
            non_zero_indices = np.nonzero(input_array[i])
            if len(non_zero_indices) > 1:
                pulse_amplitude_max = np.max(non_zero_indices)

                # calculate the pulse amplitude sum
                pulse_amplitude_sum = np.sum(
                    non_zero_indices * input_array[i][non_zero_indices]
                )

                # calculate the pulse amplitude mean
                pulse_amplitude_mean = pulse_amplitude_sum / pulse_count
            else:
                pulse_amplitude_max = 0
                pulse_amplitude_sum = 0
                pulse_amplitude_mean = 0

            basic_descriptor.append(
                {
                    "pulse_count": pulse_count,
                    "amplitude_max": pulse_amplitude_max,
                    "amplitude_sum": pulse_amplitude_sum,
                    "amplitude_mean": pulse_amplitude_mean,
                }
            )

        return basic_descriptor

    def skewness_cycle(self, input_array: np.ndarray) -> int:
        """
        Calculate the skewness of the input array.
        """
        # check the input is whether valid
        if input_array.ndim != 2:
            raise ValueError(
                "During skewness cycle feature extract,Input array must have 2 dimensions"
            )

        # calculate the mean of the input array
        mean = np.mean(input_array)

        # calculate the standard deviation of the input array
        std = np.std(input_array)

        # calculate the skewness of the input array
        skewness = 0
        skewness = np.sum((input_array - mean) ** 3) / (len(input_array) * std**3)

        return skewness

    def kurtosis_cycle(self, input_array: np.ndarray) -> int:
        """
        Calculate the kurtosis of the input array.
        """
        # check the input is whether valid
        if input_array.ndim != 2:
            raise ValueError(
                "During kurtosis cycle feature extract,Input array must have 2 dimensions"
            )

        # calculate the mean of the input array
        mean = np.mean(input_array)

        # calculate the standard deviation of the input array
        std = np.std(input_array)

        # calculate the kurtosis of the input array
        kurtosis = 0
        kurtosis = np.sum((input_array - mean) ** 4) / (len(input_array) * std**4)

        return kurtosis

    def peak_num(self, input_array: np.ndarray) -> int:
        """
        Calculate the peak num of the input array.
        """
        # check the input is whether valid
        if input_array.ndim != 2:
            raise ValueError(
                "During peak num feature extract,Input array must have 2 dimensions"
            )

        peak_count = 0

        # calculate the mean magnitude of every phase window
        mean_magnitude = np.zeros(input_array.shape[0])
        for i in range(input_array.shape[0]):
            non_zero_indices = np.nonzero(input_array[i])
            # calculate the pulse amplitude sum
            single_amplitude_sum = np.sum(
                non_zero_indices * input_array[i][non_zero_indices]
            )

            # pulse count in specified phase window
            pulse_count = np.sum(input_array[i])
            # update the mean magnitude of specified phase window
            if pulse_count != 0:
                mean_magnitude[i] = single_amplitude_sum / pulse_count
            else:
                mean_magnitude[i] = 0

        # calculate the gradient of the mean magnitude
        gradient = np.gradient(mean_magnitude)
        for i in range(1, len(gradient) - 1):
            if gradient[i - 1] > 0 and gradient[i + 1] < 0:
                peak_count += 1

        return peak_count


class ManmadeExtractorTensor:
    """
    Extract the manmade features from the input tensor.
    """

    def __init__(self) -> None:
        pass

    def pulse_phase_descriptor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract the phase descriptor of pulses of dimensions specified in the input array.

        include pulse_sum ,pulse_amplitude_max,pulse_amplitude_sum,pulse_amplitude_mean
        """

        # check the input is whether valid
        if input_tensor.ndim != 2:
            raise ValueError(
                "During pulse phase window feature extract,Input tensor must have 2 dimensions"
            )

        feature_list = []
        for i in range(input_tensor.shape[0]):
            # calculate the pulse count
            pulse_count = torch.sum(input_tensor[i])

            # calculate the pulse amplitude max
            non_zero_indices = torch.nonzero(input_tensor[i], as_tuple=False)
            if len(non_zero_indices) > 1:
                pulse_amplitude_max = torch.max(non_zero_indices).to(
                    input_tensor.device
                )

                # calculate the pulse amplitude sum
                pulse_amplitude_sum = torch.sum(
                    non_zero_indices * input_tensor[i][non_zero_indices].float()
                ).to(input_tensor.device)

                # calculate the pulse amplitude mean
                pulse_amplitude_mean = pulse_amplitude_sum / pulse_count
            else:
                pulse_amplitude_max = torch.tensor(0.0).to(input_tensor.device)
                pulse_amplitude_sum = torch.tensor(0.0).to(input_tensor.device)
                pulse_amplitude_mean = torch.tensor(0.0).to(input_tensor.device)

            descriptor_tensor = torch.tensor(
                [
                    pulse_count,
                    pulse_amplitude_max,
                    pulse_amplitude_sum,
                    pulse_amplitude_mean,
                ],
                device=input_tensor.device,
            )

            # Append the descriptor tensor to the list
            feature_list.append(descriptor_tensor)

        # Stack all descriptors into a final tensor with shape [num_samples, 4]
        feature_tensor = torch.stack(feature_list)

        return feature_tensor

    def skewness_cycle(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the skewness of the input array.
        """
        # check the input is whether valid
        if input_tensor.ndim != 2:
            raise ValueError(
                "During skewness cycle feature extract,Input tensor must have 2 dimensions"
            )

        # calculate the mean of the input array
        mean = torch.mean(input_tensor)

        # calculate the standard deviation of the input array
        std = torch.std(input_tensor)

        # Add a small epsilon to std to avoid division by zero
        epsilon = 1e-8
        std = std + epsilon

        # calculate the skewness of the input array
        skewness = torch.sum((input_tensor - mean) ** 3) / (
            input_tensor.numel() * std**3
        )

        return skewness

    def kurtosis_cycle(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the kurtosis of the input array.
        """
        # check the input is whether valid
        if input_tensor.ndim != 2:
            raise ValueError(
                "During kurtosis cycle feature extract,Input tensor must have 2 dimensions"
            )

        # calculate the mean of the input array
        mean = torch.mean(input_tensor)

        # calculate the standard deviation of the input array
        std = torch.std(input_tensor)

        # Add a small epsilon to std to avoid division by zero
        epsilon = 1e-8
        std = std + epsilon

        # calculate the kurtosis of the input array
        kurtosis = torch.sum((input_tensor - mean) ** 4) / (
            input_tensor.numel() * std**4
        )

        return kurtosis

    def peak_num(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the peak num of the input array.
        """
        # check the input is whether valid
        if input_tensor.ndim != 2:
            raise ValueError(
                "During peak num feature extract,Input tensor must have 2 dimensions"
            )

        peak_count = 0

        # Calculate the mean magnitude of every phase window
        mean_magnitude = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
        for i in range(input_tensor.shape[0]):
            # Get non-zero indices
            non_zero_indices = torch.nonzero(input_tensor[i], as_tuple=False).squeeze()

            # Calculate the pulse amplitude sum
            single_amplitude_sum = torch.sum(
                input_tensor[i][non_zero_indices] * non_zero_indices.float()
            )

            # Pulse count in specified phase window
            pulse_count = torch.sum(input_tensor[i])

            # Update the mean magnitude of specified phase window
            if pulse_count != 0:
                mean_magnitude[i] = single_amplitude_sum / pulse_count

        prepend_tensor = torch.tensor(
            [0], dtype=mean_magnitude.dtype, device=mean_magnitude.device
        )
        # calculate the gradient of the mean magnitude
        gradient = torch.diff(
            mean_magnitude, prepend=prepend_tensor
        )  # Prepend 0 to maintain shape
        peak_count = torch.sum(
            (gradient[:-1] > 0) & (gradient[1:] < 0)
        ).item()  # Count peaks

        return torch.tensor(peak_count, device=input_tensor.device)
