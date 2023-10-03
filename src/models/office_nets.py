
import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from copy import deepcopy
from torchvision.models import resnet18


class Net_Resnet18(nn.Module):
    def __init__(self, num_classes: int, device: str = 'cuda', norm_layer=nn.BatchNorm2d, norm_layer_args=None, **kwargs) -> None:
        """ Generic Resnet18 model """
        super(Net_Resnet18, self).__init__()
        if type(norm_layer) == str:
            module_name, class_name = norm_layer.rsplit(".", 1)
            norm_layer = getattr(import_module(module_name), class_name)

            if norm_layer_args is not None:
                norm_layer = partial(norm_layer, **norm_layer_args)

        self.num_classes = num_classes
        self.device = device
        
        self.net = resnet18(num_classes=self.num_classes, norm_layer=norm_layer)
    def forward(self, x):
        return self.net(x)

class Net_Office_Simple(nn.Module):
    def __init__(self, num_classes: int, conv_norm_layer=None, fc_norm_layer=None, device: str='cuda') -> None:
        super(Net_Office_Simple, self).__init__()
        self.num_classes = num_classes
        self.device= device
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2,stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5,padding=2,stride =1)
        self.conv3 = nn.Conv2d(64, 32, 5,padding=2,stride =1)
        self.fc1 = nn.Linear(32 * 32 * 32, 2048)
        self.fc2 = nn.Linear(2048, self.num_classes)

        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn3 = nn.Identity()
        self.bn4 = nn.Identity()
        if conv_norm_layer is not None:
            if type(conv_norm_layer) == str:
                module_name, class_name = conv_norm_layer.rsplit(".", 1)
                conv_norm_layer = getattr(import_module(module_name), class_name)
            self.bn1 = conv_norm_layer(32)
            self.bn2 = conv_norm_layer(64)
            self.bn3 = conv_norm_layer(32)
        
        if fc_norm_layer is not None:
            if type(fc_norm_layer) == str:
                module_name, class_name = fc_norm_layer.rsplit(".", 1)
                fc_norm_layer = getattr(import_module(module_name), class_name)
            self.bn4 = fc_norm_layer(2048)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x


class Net_Office_AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, conv_norm_layer=None, fc_norm_layer=None, device: str='cuda') -> None:
        super(Net_Office_AlexNet, self).__init__()

        if conv_norm_layer is not None:
            if type(conv_norm_layer) == str:
                module_name, class_name = conv_norm_layer.rsplit(".", 1)
                conv_norm_layer = getattr(import_module(module_name), class_name)
        else:
            conv_norm_layer = nn.Identity     

        if fc_norm_layer is not None:
            if type(fc_norm_layer) == str:
                module_name, class_name = fc_norm_layer.rsplit(".", 1)
                fc_norm_layer = getattr(import_module(module_name), class_name)
        else:
            fc_norm_layer = nn.Identity

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = conv_norm_layer(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = conv_norm_layer(192)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = conv_norm_layer(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = conv_norm_layer(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = conv_norm_layer(256)
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc_bn1 = fc_norm_layer(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc_bn2 = fc_norm_layer(4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features
        x = self.max_pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.max_pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.max_pool5(F.relu(self.bn5(self.conv5(x))))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # classifier
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

