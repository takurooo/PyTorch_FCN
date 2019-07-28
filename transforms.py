# -------------------------------------------
# import
# -------------------------------------------
import os
import re
import codecs
import random

from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
# -------------------------------------------
# defines
# -------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

# -------------------------------------------
# private functions
# -------------------------------------------

# -------------------------------------------
# public functions
# -------------------------------------------


class ResizeAndPadding:
    def __init__(self, size, interpolation=Image.BILINEAR, pad=True):
        self.size = size
        self.interpolation = interpolation
        self.pad = pad

    def __call__(self, img):
        """Resize and Padding
         # Arguments
                img: pil img, 3 dimention(H x W x C)
        """
        size_wh = img.size

        ratio = float(self.size)/max(size_wh)
        new_size_wh = tuple([int(x * ratio) for x in size_wh])

        img = img.resize(new_size_wh, resample=self.interpolation)

        if self.pad:
            mode = img.mode
            new_img = Image.new(mode, (self.size, self.size))
            new_img.paste(img,
                          ((self.size-new_size_wh[0])//2,
                           (self.size-new_size_wh[1])//2))
        else:
            new_img = img

        return new_img


class RandomCropForSemseg:
    def __init__(self, size):
        self.target_size = size

    def __call__(self, img, target=None):
        """Random crop img and label
         # Arguments
                img: pil img, 3 dimention(H x W x C)
                target: label img, 3 dimention(H x W x C)
        """
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.target_size, self.target_size))
        img = TF.crop(img, i, j, h, w)
        if target is not None:
            target = TF.crop(target, i, j, h, w)

        return img, target


class RandomFlipForSemseg:
    def __init__(self):
        pass

    def __call__(self, img, target=None):
        """Random flip img and label
         # Arguments
                img: pil img, 3 dimention(H x W x C)
                target: label img, 3 dimention(H x W x C)
        """
        if random.random() < 0.5:
            img = TF.hflip(img)
            if target is not None:
                target = TF.hflip(target)

        return img, target


# -------------------------------------------
# main
# -------------------------------------------

if __name__ == '__main__':
    pass
