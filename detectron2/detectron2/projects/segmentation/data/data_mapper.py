from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from fvcore.transforms.transform import Transform, TransformList

from .data_sample import ImageSample, SegImageSample, DefaultSegmentationSampleBuilder



class DefaultSegmentationDataMapper:
    def __init__(
            self,
            sample_builder: Callable[[Dict[str, Any]], ImageSample] = DefaultSegmentationSampleBuilder,
            transforms: Optional[List[Transform]] = None,
            sample_postprocessor: Callable[[ImageSample], None] = None
    ):
        self.transforms = TransformList(transforms) if transforms is not None else None
        self.sample_builder = sample_builder
        self.sample_postprocessor = sample_postprocessor

        def __call__(self, data: dict) -> SegImageSample:
            sample = self.sample_builder(data)

            if self.transfroms is not None:
                self.transfroms(sample)

            if self.sample_postprocessor is not None:
                self.sample_postprocessor(sample)

            return SegImageSample.from_image_sample(sample).validate()


