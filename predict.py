# -------------------------------------------
# import
# -------------------------------------------
import os
import sys
import argparse
import json

import matplotlib.pyplot as plt
import torch

from dataset import SemSegDataset
from utils import convert
from models import build_encorder_resnet18, build_fcn
# -------------------------------------------
# defines
# -------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))
JSON_PATH = os.path.join(CUR_PATH, 'args.json')

N_CLASS = 21
INPUT_SIZE = 224
# -------------------------------------------
# private functions
# -------------------------------------------


def get_args():
    with open(JSON_PATH, "r") as f:
        j = json.load(f)
    return j['predict']


def show_imgs(imgs, size, save_path=None, figsize=[6.4, 4.8], converter=False):
    """Show images
     # Arguments
            imgs: 1-D array, 4 dimention(batch x H x W x C)
            row: Int, row for plt.subplot
            col: Int, column for plt.subplot
    """
    row, col = size
    if len(imgs) != (row * col):
        raise ValueError(
            "Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))

    plt.figure(figsize=figsize)
    plt.tight_layout()

    for i, img in enumerate(imgs):
        if converter:
            img = converter(img)
        plot_num = i+1
        plt.subplot(row, col, plot_num)
        plt.axis('off')
        plt.imshow(img)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def predict(model, device, imgs):
    model.eval()
    if len(imgs.size()) != 4:
        imgs = imgs.unsqueeze(0)
    imgs = imgs.to(device)

    with torch.no_grad():
        output = model(imgs)

    return output


def main(args):
    model_name = args["model"]
    img_dir = args["img_dir"]
    log_dir = args["log_dir"]
    weight_path = args["weight_path"]
    img_idx = 30

    os.makedirs(log_dir, exist_ok=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    '''
    Create DataSet
    '''
    test_dataset = SemSegDataset(N_CLASS, INPUT_SIZE, img_dir, train=False)

    '''
    Create Model
    '''
    encorder = build_encorder_resnet18()
    model = build_fcn(model_name, num_classes=21, encorder=encorder)
    # print(model)

    '''
    Load Weight
    '''
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print("not exists weight : ", weight_path)

    '''
    Predict
    '''
    img, _ = test_dataset[img_idx]
    model.to(device)
    out = predict(model, device, img)

    '''
    Save model state_dict
    '''
    out_pil = convert.pred_to_pil(out[0])
    input_pil = convert.tensor_to_pil(img)
    img_list = [input_pil, out_pil]
    show_imgs(img_list, [1, 2],
              save_path="predict_{}.png".format(img_idx),
              figsize=(10, 5))


if __name__ == '__main__':
    main(get_args())
