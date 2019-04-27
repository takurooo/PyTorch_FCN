#-------------------------------------------
# import
#-------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision import models
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
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Make a 2D bilinear kernel suitable for upsampling
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()


def build_encorder_resnet18():
    encorder = EncorderResNet(layers=[2, 2, 2, 2])
    encorder.load_state_dict(models.resnet18(pretrained=True).state_dict())
    return encorder


def build_fcn(model_name, num_classes, encorder):
    if model_name == 'FCN32s':
        model = FCN32s(num_classes, encorder)
    elif model_name == 'FCN16s':
        model = FCN16s(num_classes, encorder)
    elif model_name == 'FCN8s':
        model = FCN8s(num_classes, encorder)
    else:
        raise ValueError('not found model_name : ', model_name)
    return model


class EncorderResNet(ResNet):
    
    def __init__(self, layers=[2, 2, 2, 2]):
        super().__init__(BasicBlock, layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = x
        x = self.layer3(x)
        x4 = x
        x = self.layer4(x)
        x5 = x
        
        return x3, x4, x5


class FCN32s(nn.Module):
    
    def __init__(self, num_classes, encorder):
        super().__init__()
        
        self.encorder = encorder
        
        self.relu = nn.ReLU(inplace=True)
        
        self.score5 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=64, stride=32, padding=16,
                                        bias=False)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init_w = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(init_w)
                
    def forward(self, x):
        
        x3, x4, x5 = self.encorder(x)
               
        x = self.score5(x5)
        
        x = self.upscore(x)
        
        return x


class FCN16s(nn.Module):
    
    def __init__(self, num_classes, encorder):
        super().__init__()
        
        self.encorder = encorder
           
        self.score4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score5 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=False)
        self.upscore5 = nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=8, stride=4, padding=2,
                                        bias=False)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=16, stride=8, padding=4,
                                        bias=False)

        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init_w = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(init_w)
    
    def forward(self, x):
        
        x3, x4, x5 = self.encorder(x)
                
        x4 = self.score4(x4)
        x5 = self.score5(x5)
        
        x4 = self.upscore4(x4)
        x5 = self.upscore5(x5)
        
        x = x4 + x5
        
        x = self.upscore(x)
        
        return x


class FCN8s(nn.Module):
    
    def __init__(self, num_classes, encorder):
        super().__init__()
        
        self.encorder = encorder
           
        self.score3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.score4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score5 = nn.Conv2d(512, num_classes, kernel_size=1)

        self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=False)
        self.upscore5 = nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=8, stride=4, padding=2,
                                        bias=False)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=16, stride=8, padding=4,
                                        bias=False)

        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init_w = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(init_w)
        
    def forward(self, x):
        
        x3, x4, x5 = self.encorder(x)
                
        x3 = self.score3(x3)
        x4 = self.score4(x4)
        x5 = self.score5(x5)
        
        x4 = self.upscore4(x4)
        x5 = self.upscore5(x5)
        
        x = x3 + x4 + x5
        
        x = self.upscore(x)
        
        return x
#-------------------------------------------
# main
#-------------------------------------------

if __name__ == '__main__':
    pass
