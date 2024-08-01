import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dataclasses import dataclass
from detectron2.utils import infer_image_shape, load_from_numpy_file, numpy_to_tensor


@dataclass
class ImageSample:
    image: Union[np.ndarray,torch.Tensor]
    img_name: Optional[str] = None
    label: Optional[Union[np.ndarray,torch.Tensor]] = None
    crop_bbox: Optional[np.ndarray] = None
    pred: Optional[torch.Tensor] = None
    x: List[torch.Tensor] = None
    y: List[torch.Tensor] = None
    logits: List[torch.Tensor] = None
    cam: Optional[Union[np.ndarray,torch.Tensor]] = None

    @property
    def img_size(self) -> Tuple[int]:
        return infer_image_shape(self.image)

@dataclass
class SegImageSample:
    image: Optional[torch.Tensor] = None
    img_name: Optional[str] = None
    label: Optional[torch.Tensor] = None
    pred: Optional[torch.Tensor] = None

    @property
    def img_size(self) -> Tuple[int, ...]:
        return infer_image_shape(self.image)

    @classmethod
    def from_image_sample(cls, sample: ImageSample):
        return cls(
            image=numpy_to_tensor(sample.image),
            label=numpy_to_tensor(sample.label.astype(np.int64)),
            img_name=sample.image,
        )

    def validate(self):
        return self


class DefaultSegmentationSampleBuilder:
    def __init__(
        self,
        image_loader: Callable[[str], np.ndarray] = load_from_numpy_file,
        label_loader: Callable[[str], np.ndarray] = load_from_numpy_file,
    ):
        self.image_loader = image_loader
        self.label_loader = label_loader

    def __call__(self, data: Dict[str, Any]) -> ImageSample:
        image = self.image_loader(data['img_path']).astype(np.float32)
        label = self.label_loader(data['lab_path']).astype(np.uint8)
        return ImageSample(
            image=image,
            img_name=data['img_name'],
            label=label
        )