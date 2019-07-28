# -------------------------------------------
# import
# -------------------------------------------
import os
import sys
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SemSegDataset
from models import build_encorder_resnet18, build_fcn
from trainer import Trainer
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
    return j['train']


def main(args):
    model_name = args["model"]
    train_img_dir = args["train_img_dir"]
    train_gt_dir = args["train_gt_dir"]
    val_img_dir = args["val_img_dir"]
    val_gt_dir = args["val_gt_dir"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    log_dir = args["log_dir"]

    print("make logdir : ", log_dir)
    os.makedirs(log_dir, exist_ok=True)

    '''
    Create DataLoader
    '''
    print("make train dataset")
    print(train_img_dir)
    print(train_gt_dir)
    train_dataset = SemSegDataset(N_CLASS,
                                  INPUT_SIZE,
                                  train_img_dir,
                                  train_gt_dir,
                                  train=True)
    print("make val dataset")
    print(val_img_dir)
    print(val_gt_dir)
    val_dataset = SemSegDataset(N_CLASS,
                                INPUT_SIZE,
                                val_img_dir,
                                val_gt_dir,
                                train=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=24, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=24, shuffle=False, num_workers=4)

    steps_per_epoch = len(train_dataset) // batch_size
    # validation_steps = len(val_dataset) // batch_size

    print("train_img_len   : {}".format(len(train_dataset)))
    print("val_img_len     : {}".format(len(val_dataset)))
    print("epochs          : ", epochs)
    print("batch_size      : ", batch_size)
    print("steps_per_epoch : ", steps_per_epoch)

    '''
    Create Model
    '''
    print("build model")
    encorder = build_encorder_resnet18()
    model = build_fcn(model_name, num_classes=21, encorder=encorder)
    # print(model)

    '''
    Setting loss and optimzer, device
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''
    Start Training
    '''
    model = model.to(device)
    trainer = Trainer(model, device, optimizer, criterion,
                      train_dataloader, val_dataloader)

    print("start training...")
    trainer.train_loop(epochs)

    '''
    Save model state_dict
    '''
    import datetime
    now = datetime.datetime.now()
    save_model_path = os.path.join(
        log_dir, 'fcn_{:%Y%m%d_%H%M%S}.pt'.format(now))
    trainer.save_best_model(save_model_path)
    print("save : ", save_model_path)


if __name__ == '__main__':
    main(get_args())
