#-------------------------------------------
# import
#-------------------------------------------
import os
import re
import codecs
from PIL import Image
import numpy as np
from utils import list_util
from utils import convert

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
#-------------------------------------------
# defines
#-------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

#-------------------------------------------
# private functions
#-------------------------------------------

#-------------------------------------------
# public functions
#-------------------------------------------

class SemSegDataset(Dataset):
    
    def __init__(self, classes, input_size, img_dir, target_dir=None, train=False):
        self.classes = classes
        self.input_size = input_size
        
        self.img_paths = list_util.list_from_dir(img_dir, ('.jpg', '.png'))
        if target_dir:
            self.target_paths = list_util.list_from_dir(target_dir, ('.jpg', '.png'))
        else:
            self.target_paths = None
            
        self.train = train
        
        self.eval_trns = transforms.Compose([
            transforms.CenterCrop(self.input_size)
        ])

        # check data num
        img_num = len(self.img_paths)
        if target_dir:
            target_num = len(self.target_paths)
            assert img_num == target_num, "img/gt not equal, img:{} gt:{}".format(img_num, target_num)
        
        self.data_num = img_num

        
    def __len__(self):
        return self.data_num
    
    
    def _transform(self, img, target, train):
        
        if train:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.input_size)
            img = TF.crop(img, i, j, h, w)
            if target:
                target = TF.crop(target, i, j, h, w)
      
            if random.random() < 0.5:
                image = TF.hflip(image)
                if target:
                    target = TF.hflip(target)
            
        else:
            img = self.eval_trns(img)
            if target:
                target = self.eval_trns(target)
     
        img = convert.pil_to_tensor(img) # 0-255 -> 0-1
        
        if target:
            target = np.asarray(target, dtype=np.int32)
            target[target == 255] = 0  # 境界部分をbackgroundクラスにする
            target = convert.pil_to_one_hot_array(target, self.classes, self.input_size)
            target = torch.from_numpy(target).long() # ラベルをtensor型に変換
        
        return img, target

    def __getitem__(self, idx):
        assert idx < self.data_num,"invalid idx {}, max idx is {}".format(idx, self.data_num-1)
        
        img = Image.open(self.img_paths[idx])
        if self.target_paths:
            target = Image.open(self.target_paths[idx])
        else:
            target = None
        
        x, y = self._transform(img, target, self.train)

        return x, y


#-------------------------------------------
# main
#-------------------------------------------

if __name__ == '__main__':
    pass
