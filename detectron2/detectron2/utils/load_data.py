from typing import Union, List, Dict
import numpy as np
import torch

def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.ascontiguousarray(arr)
    return torch.as_tensor(arr)

def infer_image_shape(image: Union[np.ndarray, torch.Tensor]):
    img_shape = image.shape
    assert len(img_shape) in [3, 4]
    if len(img_shape) ==3:
        return img_shape[-2:]
    else:
        return img_shape[-3:]

def load_from_numpy_file(file_path: str, pad_channel_dim: bool = True, lazy_load: bool = False):
    data = np.load(file_path, mmap_mode='r' if lazy_load else None)
    if isinstance(data, np.lib.npyio.NpzFile):
        data = data['arr_0']

    if pad_channel_dim and len(data.shape) == 3:
        data = data[None]

    return data