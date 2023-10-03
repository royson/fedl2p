import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from functools import partial
from torchvision.models import resnet18


class Net_Commands_Resnet18(nn.Module):
    def __init__(self, num_classes: int, device: str = 'cuda', norm_layer=nn.BatchNorm2d, norm_layer_args=None, **kwargs) -> None:
        """ A ResNet18 adapted to SpeechCommands. """
        super(Net_Commands_Resnet18, self).__init__()
        if type(norm_layer) == str:
            module_name, class_name = norm_layer.rsplit(".", 1)
            norm_layer = getattr(import_module(module_name), class_name)

            if norm_layer_args is not None:
                norm_layer = partial(norm_layer, **norm_layer_args)

        self.num_classes = num_classes
        self.device = device
        
        self.net = resnet18(num_classes=self.num_classes, norm_layer=norm_layer)
        # replace w/ smaller input layer
        # self.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        nn.init.kaiming_normal_(self.net.conv1.weight, mode='fan_out', nonlinearity='relu')
        # no need for pooling if training for CIFAR-10
        self.net.maxpool = torch.nn.Identity()
    def forward(self, x):
        return self.net(x)

