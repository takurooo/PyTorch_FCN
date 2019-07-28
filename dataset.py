# -------------------------------------------
# import
# -------------------------------------------
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from utils.list_util import *
from utils.convert import *
from transforms import *
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


class SemSegDataset(Dataset):

    def __init__(self, classes, input_size, img_dir, target_dir=None, train=False):
        self.classes = classes
        self.input_size = input_size
        self.img_paths, self.target_paths = self._get_paths(
            img_dir, target_dir)
        self.train = train

        # -------------------------------------------
        # transforms
        # -------------------------------------------
        self.trans_resize_and_pad = ResizeAndPadding(self.input_size)
        self.train_trans_crop = RandomCropForSemseg(self.input_size)
        self.train_trans_flip = RandomFlipForSemseg()
        self.eval_trns = transforms.Compose([
            transforms.CenterCrop(self.input_size)
        ])

        # -------------------------------------------
        # check data num
        # -------------------------------------------
        img_num = len(self.img_paths)
        if target_dir:
            target_num = len(self.target_paths)
            assert img_num == target_num, "img/gt not equal, img:{} gt:{}".format(
                img_num, target_num)

        self.data_num = img_num

    def __len__(self):
        return self.data_num

    def _get_paths(self, img_dir, target_dir):
        img_paths = list_from_dir(img_dir, ('.jpg', '.png'))
        if target_dir:
            target_paths = list_from_dir(target_dir, ('.jpg', '.png'))
        else:
            target_paths = None

        return img_paths, target_paths

    def _transform(self, img, target, train):

        # -------------------------------------------
        # Padding and Resize if needed
        # -------------------------------------------
        if min(img.size) < self.input_size:
            img = self.trans_resize_and_pad(img)
            if target:
                target = self.trans_resize_and_pad(target)

        if train:
            # -------------------------------------------
            # Random Crop
            # -------------------------------------------
            img, target = self.train_trans_crop(img, target)

            # -------------------------------------------
            # Random Flip
            # -------------------------------------------
            img, target = self.train_trans_flip(img, target)

        else:
            # -------------------------------------------
            # Evaluation
            # -------------------------------------------
            img = self.eval_trns(img)
            if target:
                target = self.eval_trns(target)

        img = pil_to_tensor(img)  # 0-255 -> 0-1

        if target:
            target = np.asarray(target, dtype=np.int32)
            target[target == 255] = 0  # 境界部分をbackgroundクラスにする

            target = pil_to_one_hot_array(
                target, self.classes, self.input_size)
            target = torch.from_numpy(target).long()  # ラベルをtensor型に変換

        return img, target

    def __getitem__(self, idx):
        assert idx < self.data_num, "invalid idx {}, max idx is {}".format(
            idx, self.data_num-1)

        img = Image.open(self.img_paths[idx])
        if self.target_paths:
            target = Image.open(self.target_paths[idx])
        else:
            target = None

        x, y = self._transform(img, target, self.train)

        return x, y


# -------------------------------------------
# main
# -------------------------------------------

if __name__ == '__main__':
    pass
