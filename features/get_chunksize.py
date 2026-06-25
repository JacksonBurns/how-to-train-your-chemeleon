import numpy as np

DEFAULT = 146  # can set to a fixed value, particularly for multi-task training when matched size is needed


def get_chunk_rows(dtype, descriptor_count):
    if DEFAULT is not None:
        return DEFAULT
    # Estimate chunk size for ~1MB chunks
    # other suggestions here: https://github.com/zarr-developers/zarr-python/issues/86#issuecomment-254439393
    bytes_per_value = np.dtype(dtype).itemsize
    return (1 * 1024 * 1024) // (descriptor_count * bytes_per_value)
