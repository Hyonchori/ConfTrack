# util functions for dataset loading

from typing import List

import cv2
import numpy as np


def preproc(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img


class InferenceTransform:
    def __init__(
            self,
            img_size: List[int] = (640, 640),
            rgb_means: List[float] = (0.485, 0.456, 0.406),
            rgb_std: List[float] = (0.229, 0.224, 0.225),
            swap: List[int] = (2, 0, 1)
    ):
        self.img_size = img_size
        self.rgb_means = rgb_means
        self.rgb_std = rgb_std
        self.swap = swap

    def __call__(self, img: np.ndarray, origin_size: bool = False):
        # resize and padding
        if origin_size:
            img = img.copy().astype(np.float64)
        else:
            img = preproc(img, self.img_size)

        # bgr2rgb
        img = img[:, :, ::-1]

        # scaling [0 ~ 255] -> [0.0 ~ 1.0]
        img /= 255.0

        # normalize
        img -= self.rgb_means
        img /= self.rgb_std

        # swap channels [height, width, channels] -> [channels, height, width]
        img = img.transpose(self.swap)

        # make contiguous array form faster inference
        img = np.ascontiguousarray(img, np.float32)

        return img
