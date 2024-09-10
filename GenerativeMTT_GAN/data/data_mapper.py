from __future__ import annotations

import numpy as np
import matplotlib.image as mpimg
from typing import List

from detectron2.projects.generation.data.data_sample import ImageSample
from detectron2.utils import numpy_to_tensor
from detectron2.projects.segmentation.transforms import Transform, TransformList


class GnrtMTTGANDataMapper:
    def __init__(self,
                 transforms: List[Transform],):

        self.transforms = TransformList(transforms)

    def __call__(self, data: dict) -> ImageSample:
        # RGB = mpimg.imread(data['RGB_image_path']).astype(np.float32)/255
        # RGB = np.transpose(RGB, axes=(2,0,1))
        LSI = mpimg.imread(data['LSI_image_path']).astype(np.float32)[None]/255
        MTT = mpimg.imread(data['MTT_image_path']).astype(np.float32)[None]/255
        v_mask = ((mpimg.imread(data['vessel_mask_path']).astype(np.float32)[None] / 255)>0.5).astype(np.uint8)
        img_name = data['case_name']

        # image = np.concatenate([LSI, RGB], 0)

        sample = ImageSample(
            img_name = img_name,
            image_A = LSI,
            image_B = MTT,
            v_mask=v_mask
        )

        if len(self.transforms)>0:
            self.transforms(sample)

        return ImageSample(
            img_name=sample.img_name,
            image_A=numpy_to_tensor(sample.image_A),
            image_B=numpy_to_tensor(sample.image_B),
            v_mask=numpy_to_tensor(sample.v_mask)
        )




