#-------------------------------------------
# import
#-------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
#-----------------------------------------
# defines
#-----------------------------------------

#-----------------------------------------
# functions
#-----------------------------------------
def make_cmap():
    ctbl = ((0x80, 0, 0), (0, 0x80, 0), (0, 0, 0x80),
            (0x40, 0, 0), (0, 0x40, 0), (0, 0, 0x40),
            (0x20, 0, 0), (0, 0x20, 0))

    n = 256
    lookup = np.zeros((n, 3)).astype(np.int32)
    for i in range(0, n):
        r, g, b = 0, 0, 0
        for j in range(0, 7):
            bit = (i >> j) & 1
            if bit:
                r |= ctbl[j][0]
                g |= ctbl[j][1]
                b |= ctbl[j][2]

        lookup[i, 0], lookup[i, 1], lookup[i, 2] = r, g, b
    return lookup[0:21]


def pil_to_tensor(pil_img):
    """ Convert from PIL to tensor
     # Arguments
            pil_img: array, 3 dimention(H x W x C)
    """
    img = np.array(pil_img, copy=False)
    img = img.transpose((2, 0, 1)) # H x W x C -> C x H x W
    img = img.astype(np.float32) # uint8 -> float32
    img = torch.from_numpy(img) # ndarray -> tensor
    img.div_(255)
    return img


def tensor_to_pil(tesnor_img):
    """ Convert from Tensor to PIL
     # Arguments
            tesnor_img: tensor, 3 dimention(C x H x W )
    """
    img = tesnor_img.mul(255).numpy() # tensor -> ndarray
    img = np.array(img, dtype=np.uint8) # float32 -> uint8
    img = img.transpose(1,2,0) # C x H x W -> H x W x C
    return Image.fromarray(img)


def tensor_to_ndarray(tesnor_img):
    """ Convert from Tensor to ndarray
     # Arguments
            tesnor_img: tensor, 3 dimention(C x H x W )
    """
    img = tesnor_img.mul(255).numpy() # tensor -> ndarray
    img = np.array(img, dtype=np.uint8) # float32 -> uint8
    img = img.transpose(1,2,0) # C x H x W -> H x W x C
    return img



def pil_to_one_hot_array(label_array, classes, size):
    """ Convert from PIL to one-hot-array
     # Arguments
            label_array: array, 3 dimention(H x W x C)
            classes: int, num of class
            size: int, image shape
    """
    x = np.zeros((classes, size, size)) # C x H x W
    for i in range(size):
        for j in range(size):
            x[label_array[i][j], i, j] = 1
    return x


def pred_to_pil(pred):
    """ Convert from one-hot-array to PIL
     # Arguments
            pred: tensor, 3 dimention(C x H x W)
    """
    cmap = make_cmap()
    
    pred_ = pred.numpy() # tensor -> ndarray
    channel, _, _ = pred_.shape
    pred_ = pred_.transpose(1, 2, 0) # C x H x W -> H x W x C
    pred_ = np.argmax(pred_, axis=2) #  H x W x C -> H x W
    
    row, col = pred_.shape
    dst = np.ones((row, col, 3))
    
    for i in range(channel):
        dst[pred_ == i] = cmap[i]

    dst = dst.astype(np.uint8)
    return  Image.fromarray(dst) # ndarray -> PIL

#-----------------------------------------
# main
#-----------------------------------------
if __name__ == '__main__':
    pass
