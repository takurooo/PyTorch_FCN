# Fully Convolutional Networks for semantic segmentation

PyTorch implementation of 
[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038).

# Usage
## Training
1 Create a new folder and put training and validation data.  
2 Write training config in "args.json".  
3 `python train.py`  
4 Start training.

## Prediction
1 Create a new folder and put test data.  
2 Write predict config in "args.json".  
3 `python predict.py`  
4 Start prediction and show result images.

args.json
```json
{
    "train": {
        "model": "FCN8s",
        "train_img_dir": "data/train/img",
        "train_gt_dir": "data/train/gt",
        "val_img_dir": "data/val/img",
        "val_gt_dir": "data/val/gt",
        "epochs": 1,
        "batch_size": 24,
        "log_dir": "log"
    },
    "predict": {
        "model": "FCN8s",
        "img_dir": "data/val/img",
        "log_dir": "log",
        "weight_path": "log/fcn8s.pt"
    }
}
```


# Results
Left : Input image  
Center : GT  
Right : Predicted image  
![prediction1](https://user-images.githubusercontent.com/35373553/56850392-f0c66980-693c-11e9-9aed-8160200cf2a7.png)

![prediction2](https://user-images.githubusercontent.com/35373553/56850394-f15f0000-693c-11e9-9682-f54c0d5ef9a5.png)

![prediction3](https://user-images.githubusercontent.com/35373553/56850396-f15f0000-693c-11e9-8a5e-effd31c5face.png)

![prediction4](https://user-images.githubusercontent.com/35373553/56850397-f15f0000-693c-11e9-8d85-75dd487b4276.png)


## accuracy(PASCAL VOC2012)
![acc](https://user-images.githubusercontent.com/35373553/56850528-d4c3c780-693e-11e9-9ee4-effc3d63ef65.png)

## loss(PASCAL VOC2012)
![loss](https://user-images.githubusercontent.com/35373553/56850527-d4c3c780-693e-11e9-8ae9-fdf28eb1b4c2.png)