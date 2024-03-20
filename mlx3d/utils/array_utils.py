import mlx.core as mx


def unique_num_items(a: mx.array):
    uniques_ = {}
    for item in a:
        if not item in uniques_:
            uniques_[item] = 1

    return len(uniques_)
