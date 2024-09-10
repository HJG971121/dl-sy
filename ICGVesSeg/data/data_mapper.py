from __future__ import annotations

import numpy as np
import matplotlib.image as mpimg
from typing import List
import scipy.ndimage as ndimage
from detectron2.utils import numpy_to_tensor
from detectron2.projects.segmentation.data import ImageSample
from detectron2.projects.segmentation.transforms import Transform, TransformList


class VesSegDataMapper:
    def __init__(self,
                 transforms: List[Transform],):

        self.transforms = TransformList(transforms)

    def __call__(self, data: dict) -> ImageSample:
        image = mpimg.imread(data['image_path']).astype(np.float32)[None]/255
        label = mpimg.imread(data['mask_path']).astype(np.float32)[None] / 255
        img_name = data['case_name']

        sample = ImageSample(
            img_name = img_name,
            image = image,
            label = label,
        )

        if len(self.transforms)>0:
            self.transforms(sample)

        return ImageSample(
            img_name=sample.img_name,
            image=numpy_to_tensor(sample.image),
            label=numpy_to_tensor(sample.label)
        )




