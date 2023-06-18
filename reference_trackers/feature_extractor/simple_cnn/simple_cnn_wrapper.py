# BringUp Simple_CNN for extracting feature (not for train)

import os
from typing import List
from pathlib import Path

import torch
import torchvision
import torch.nn as nn

from .reid_model import Net

FILE = Path(__file__).absolute()


class WrappedSimpleCNN(nn.Module):
    def __init__(
            self,
            num_classes: int = 751,  # number of classes in MARS dataset
            weights_file: str = None,
            input_size: List[int] = (128, 64),  # [height, width]
            device: torch.device = None,
            half: bool = False
    ):
        super().__init__()
        print('\nLoading feature_extractor "SimpleCNN"...')
        print(f'\tfeature dim: {num_classes}, img_size: {input_size}')
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.num_classes = num_classes
        self.input_size = input_size

        self.model = self._init_model(num_classes, weights_file).to(self.device)
        self.model.eval()

        self.preproc = torchvision.transforms.Resize(self.input_size)

        if half:
            self.model.half()
            print('\tHalf tensor type!')

    @staticmethod
    def _init_model(num_classes: int, weights_file: str) -> nn.Module:
        model = Net(num_classes=num_classes, reid=True)

        if weights_file is not None:
            if not os.path.isfile(weights_file):
                weights_dir = os.path.join(FILE.parents[3], 'pretrained', 'feature_extractor', 'simple_cnn')
                weights_file = os.path.join(weights_dir, weights_file)
            weights = torch.load(weights_file)['net_dict']
            model.load_state_dict(weights)
            print(f'\tpretrained extractor weights "{os.path.basename(weights_file)}" are loaded!')
        else:
            print('\tpretrained extractor weights is None.')

        return model

    def preprocessing(self, xyxys, img):  # img: torch.Tensor (batch_size, channels, height, width)
        crops = []
        for xyxy in xyxys:
            crops.append(
                self.preproc(img[:, :, max(0, int(xyxy[1])): int(xyxy[3]), max(0, int(xyxy[0])): int(xyxy[2])])
            )
        crops = torch.cat(crops).to(self.device).type_as(next(self.model.parameters()))

        return crops

    def forward(self, x, img=None):
        if img is not None:
            x = self.preprocessing(x, img)
        x = self.model(x)
        return x.cpu().data.numpy()


def get_wrapped_simple_cnn(extractor_cfg, device: torch.device = None):
    if extractor_cfg.extractor_weights == 'simple_cnn_mars':
        num_classes = 751
        input_size = [128, 64]
        weights_file = 'ckpt.t7'

    else:
        raise Exception(f'Given extractor weights "{extractor_cfg.extractor_weights}" '
                        f'is not valid for wrapped_simple_cnn!')

    return WrappedSimpleCNN(
        num_classes=num_classes,
        input_size=input_size,
        weights_file=weights_file,
        device=device,
        half=extractor_cfg.half
    )
