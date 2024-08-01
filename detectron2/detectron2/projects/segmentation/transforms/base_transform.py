from typing import Union, Set, Dict, List, Sequence, Callable, Any
import numpy as np
from ..data.data_sample import ImageSample


__all__ = [
    'Transform',
    'TransformList',
    'TransformFields'
]


TransformFields = Union[Set[str], Dict[str, str]]

class Transform:
    def __init__(self,
                 fields: TransformFields = None):

        if fields is None:
            fields = {'image'}
        self.fields = fields
        from omegaconf.dictconfig import DictConfig
        if isinstance(fields, (dict, DictConfig)):
            field_types = set(fields.values())
        else:
            field_types = fields

        for type_name in field_types:
            assert self._get_apply(type_name) is not None, f'apply_{type_name} must be specified for {type(self)}'

    def _get_apply(self, type_name: str) -> Callable:
        return getattr(self, 'apply_{}'.format(type_name), None)

    def _validate(self, sample: ImageSample) -> None:
        return None

    def _init(self, sample: ImageSample) -> None:
        pass

    def __call__(self, sample: ImageSample) -> 'Transform':
        self._init(sample)
        self._validate(sample)

        if isinstance(self.fields, set):
            for field_name in self.fields:
                field = getattr(sample, field_name)
                res = self._get_apply(field_name)(field)
                setattr(sample, field_name, res)
        else:
            for field_name, type_name in self.fields.items():
                field = getattr(sample, field_name)
                res = self._get_apply(type_name)(field)
                setattr(sample, field_name, res)
        return self

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class TransformList(List[Callable]):
    def __init__(self, transforms: Sequence[Callable]):
        super().__init__(transforms)

    def __call__(self, sample: Any) -> 'Transform':
        for transform in self:
            transform(sample)
        return self