import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from functools import partial
from torchvision.models import resnet18

class Net_Cifar_CNN(nn.Module):
    """Simple CNN model for cifar can be updated."""

    def __init__(self, num_classes: int, device: str = 'cuda', **kwargs) -> None:
        super(Net_Cifar_CNN, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_Cifar_Resnet18(nn.Module):
    def __init__(self, num_classes: int, device: str = 'cuda', norm_layer=nn.BatchNorm2d, norm_layer_args=None, **kwargs) -> None:
        """ A ResNet18 adapted to CIFAR10. """
        super(Net_Cifar_Resnet18, self).__init__()
        if type(norm_layer) == str:
            module_name, class_name = norm_layer.rsplit(".", 1)
            norm_layer = getattr(import_module(module_name), class_name)

            if norm_layer_args is not None:
                norm_layer = partial(norm_layer, **norm_layer_args)

        self.num_classes = num_classes
        self.device = device
        
        self.net = resnet18(num_classes=self.num_classes, norm_layer=norm_layer)
        # replace w/ smaller input layer
        self.net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.net.conv1.weight, mode='fan_out', nonlinearity='relu')
        # no need for pooling if training for CIFAR-10
        self.net.maxpool = torch.nn.Identity()
    def forward(self, x):
        return self.net(x)

