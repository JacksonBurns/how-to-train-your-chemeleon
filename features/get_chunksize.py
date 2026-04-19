import numpy as np

def get_chunk_rows(dtype, descriptor_count):
    # Estimate chunk size for ~1MB chunks
    # other suggestions here: https://github.com/zarr-developers/zarr-python/issues/86#issuecomment-254439393
    bytes_per_value = np.dtype(dtype).itemsize
    return (1 * 1024 * 1024) // (descriptor_count * bytes_per_value)
