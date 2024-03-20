import mlx.core as mx
import numpy as np


def unique_num_items(a: mx.array):
    # Flatten the array to 1D
    flat_array = np.array(a).flatten()

    # Use a set to find unique elements
    uniques_ = set(flat_array)

    return len(uniques_)
