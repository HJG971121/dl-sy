import numpy as np
import random
import cv2
from typing import Sequence, Tuple, Optional, Union, Literal
from abc import ABCMeta, abstractmethod
from scipy.ndimage import rotate

from .base_transform import Transform, TransformList, TransformFields
from ..data import ImageSample

class CropTransform(Transform, metaclass = ABCMeta):
    def __init__(self,
                 crop_size: Tuple[int, int],
                 fields: TransformFields = None):
        super().__init__(fields)
        assert len(crop_size) == 2, f'crop size should be 2, but got {len(crop_size)}.'

        self.crop_size = crop_size
        self.start = None

    @abstractmethod
    def set_from(self, img_size: Tuple[int, int]):
        raise NotImplementedError

    def _init(self, sample: ImageSample):
        self.set_from(sample.img_size)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return image[...,
               self.start[0]: self.start[0]+self.crop_size[0],
               self.start[1]: self.start[1]+self.crop_size[1]]

    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        boxes = boxes.reshape(-1, 2, 2) - np.array(self.start)[::-1].reshape(1,1,2)
        boxes = boxes.reshape(-1, 4)
        return boxes

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords - np.array(self.start)[::-1].reshape(1,2)


class FlipTransform(Transform, metaclass = ABCMeta):
    def __init__(self,
                 flip_axis: Tuple[int, int],
                 flip_freq: int = 50,
                 fields: TransformFields = None):
        super().__init__(fields)
        assert len(flip_axis) == 2, f'crop size should be 2, but got {len(flip_axis)}.'
        assert flip_freq<=100, f'flip frequency should less than 100, but got {flip_freq}'

        self.flip_axis = flip_axis
        self.flip_freq = flip_freq
        self.is_flip = [False, False]

    def set_from(self):
        for axis, is_flip in enumerate(self.flip_axis):
            if is_flip == 1:
                idx = random.randint(1, 100)
                if idx < self.flip_freq:
                    self.is_flip[axis] = True

    def _init(self, sample: ImageSample):
        self.set_from()

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        for axis, is_flip in enumerate(self.is_flip):
            if is_flip:
                image = np.flip(image, axis=axis+1)
        return image

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation)


class RotateTransform(Transform, metaclass = ABCMeta):
    def __init__(self,
                 rot_angle: Tuple[int, int],
                 rot_freq: int = 50,
                 fields: TransformFields = None):
        super().__init__(fields)
        assert -180<=rot_angle[0]<=rot_angle[1]<=180, f'Rotate angle range should be in [-180, 180].'
        assert rot_freq<=100, f'Rotate frequency should less than 100, but got {rot_freq}'

        self.rot_angle = rot_angle
        self.rot_freq = rot_freq

    def set_from(self, img_size: Tuple[int, int]):
        self.rot_center = tuple(np.asarray(img_size)//2)
        if random.randint(0, 100)<=self.rot_freq:
            self.angle = random.randint(self.rot_angle[0], self.rot_angle[1])
        else:
            self.angle = None
        self.img_size = img_size


    def _init(self, sample: ImageSample):
        self.set_from(sample.img_size)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if self.angle == None:
            return image
        return rotate(image, self.angle, reshape=False, axes=(1,2), mode='mirror')

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation)


class RandomCBTransform(Transform, metaclass = ABCMeta):
    def __init__(self,
                 contrast_range: Tuple[float, float]=[0.8, 1.2],
                 brightness_range: Tuple[int, int]=[-50, 50],
                 transform_freq: int = 50,
                 fields: TransformFields = None):
        super().__init__(fields)
        assert 0<=contrast_range[0]<=contrast_range[1], f'Contrast range should be larger than 0.'
        assert brightness_range[0]<=brightness_range[1], f'brightness_range[0] should be smaller than brightness_range[1].'

        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.transform_freq = transform_freq

    def set_from(self):
        if random.randint(0, 100)<=self.transform_freq:
            self.contrast_factor = round(random.uniform(self.contrast_range[0], self.contrast_range[1]),2)
            self.brightness_factor = random.randint(self.brightness_range[0], self.brightness_range[1])
        else:
            self.contrast_factor = None
            self.brightness_factor = None


    def _init(self, sample: ImageSample):
        self.set_from()

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[0]<3:
            print('The channel of input is smaller than 3, skip Contrast&Brightness transform.')
            return image
        else:
            # print(image.shape)
            RGB = np.transpose(image[-3:], axes=(1,2,0))

        if self.contrast_factor == None:
            return image

        # print(f'contrast factor: {self.contrast_factor}, brightness factor: {self.brightness_factor}.')
        if RGB.dtype != np.uint8:
            RGB = (RGB*255).astype(np.uint8)
        RGB = cv2.convertScaleAbs(RGB, alpha=self.contrast_factor, beta=self.brightness_factor)
        image[-3:] = np.transpose(RGB, axes=(2,0,1)).astype(np.float32)/255

        return image

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation



class CenterCrop(CropTransform):
    def set_from(self, img_size: Tuple[int, int]):
        img_size = np.asarray(img_size)
        crop_size = np.asarray(self.crop_size)
        assert np.all(img_size >= crop_size), \
        f"crop size {self.crop_size} should be smaller than image size {img_size}."

        start = (img_size - crop_size) //2
        self.start = tuple(start.tolist())


class RandomCrop(CropTransform):
    def set_from(self, img_size: Tuple[int, int]):
        img_size = np.asarray(img_size)
        crop_size = np.asarray(self.crop_size)
        assert np.all(img_size >= crop_size), \
            f"crop size {self.crop_size} should be smaller than image size {img_size}."

        self.start = tuple(random.randint(0, max(0, img_size[i] - self.crop_size[i]))
                           for i in range(len(crop_size)))

class RandomSizeCrop(CropTransform):
    def set_from(self, img_size: Tuple[int, int]):

        img_size = np.asarray(img_size)
        # crop_size = np.asarray([256, 256])
        # crop_stride = np.asarray(crop_size) / 16
        # crop_size = np.minimum(crop_size + crop_stride * random.randint(0, 16), img_size)
        crop_size = np.asarray([random.randint(x,512) for x in self.crop_size])
        crop_size = np.minimum(crop_size,img_size)
        # print(f'crop_size: {crop_size}')
        assert np.all(img_size >= crop_size), \
            f"crop size {self.crop_size} should be smaller than image size {img_size}."

        self.start = tuple(random.randint(0, max(0, img_size[i] - self.crop_size[i]))
                           for i in range(len(crop_size)))

class RandomCrop_pad(CropTransform):
    def __init__(self,
                 crop_size: Tuple[int, int],
                 margin: Tuple[int, int],
                 pad_value: float,
                 pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
                 fields: TransformFields = None
                 ):
        super().__init__(crop_size, fields)

        self.margin = np.asarray(margin)
        self.crop_size = np.asarray(crop_size)
        self.pad_value = pad_value
        self.pad_mode = pad_mode
        self.pad = ((0, 0),(margin[0], margin[0]), (margin[1], margin[1]))

    def set_from(self, img_size: Tuple[int, int]):
        img_size = np.asarray(img_size)
        crop_size = np.asarray(self.crop_size)
        assert np.all(img_size >= crop_size), \
            f"crop size {self.crop_size} should be smaller than image size {img_size}."

        self.start = tuple(random.randint(0, max(0, img_size[i] - self.crop_size[i]))
                           for i in range(len(crop_size)))

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        image = image[...,
               self.start[0]: self.start[0]+self.crop_size[0],
               self.start[1]: self.start[1]+self.crop_size[1]]
        if self.pad_mode == 'constant':
            image = np.pad(image, self.pad, mode=self.pad_mode, constant_values=self.pad_value)
        else:
            image = np.pad(image, self.pad, self.pad_mode)
        return image




