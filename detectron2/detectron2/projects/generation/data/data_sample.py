import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass
from detectron2.utils import infer_image_shape, load_from_numpy_file, numpy_to_tensor


@dataclass
class ImageSample:
    image_A: Union[np.ndarray, torch.Tensor]
    image_B: Union[np.ndarray, torch.Tensor] = None
    v_mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    img_name: Optional[str] = None
    paired: Optional[bool] = None
    pred: Optional[torch.Tensor] = None


    @property
    def img_size(self) -> Tuple[int]:
        return infer_image_shape(self.image_A)