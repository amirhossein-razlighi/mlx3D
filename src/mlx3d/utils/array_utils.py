import mlx.core as mx
import numpy as np


def unique_num_items(a: mx.array):
    # Flatten the array to 1D
    flat_array = np.array(a).flatten()

    # Use a set to find unique elements
    uniques_ = set(flat_array)

    return len(uniques_)


def boolean_indexing(a: mx.array, b: mx.array):
    new_array = mx.zeros((0, a.shape[1]), dtype=a.dtype)
    for i in range(a.shape[0]):
        if b[i]:
            new_array = mx.concatenate((new_array, mx.expand_dims(a[i], axis=0)))
    return new_array
