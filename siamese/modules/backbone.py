import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet, mobilenetv2

from . import classifier

class SigNetBackbone(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(3, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )

        self.fc = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # Forward pass 
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class ResNetBackbone(resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__(block, layers, num_classes)
        self.block_expansion = block.expansion

class MobileNetV2Backbone(mobilenetv2.MobileNetV2):
    def __init__(self, num_classes: int = 1000):
        super(MobileNetV2Backbone, self).__init__(num_classes=num_classes)
        

def resnet_backbone(pretrained_backbone=True, encoder_digit=64, version=18, in_chan=3, **kwargs):
    if in_chan != 3 and pretrained_backbone:
        raise ValueError("in_chan has to be 3 when you set pretrained=True")

    block = {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3], '50': [3, 4, 6, 3],
             '101': [3, 4, 23, 3], '152': [3, 8, 36, 3]}
    name_ver = 'resnet'+str(version)

    backbone_model = ResNetBackbone(resnet.BasicBlock, block[str(version)], **kwargs)
    if pretrained_backbone:
        state_dict = model_zoo.load_url(resnet.model_urls[name_ver])
        backbone_model.load_state_dict(state_dict)
    expansion = 512 * backbone_model.block_expansion
    backbone_model.fc = classifier.Classfiers(in_features=expansion, n_classes=encoder_digit)
    return backbone_model


def mobilenetv2_backbone(pretrained_backbone=True, encoder_digit=64, progress=True, **kwargs):
    backbone_model = mobilenetv2.MobileNetV2()
    if pretrained_backbone:
        state_dict = load_state_dict_from_url(mobilenetv2.model_urls['mobilenet_v2'], progress=progress)
        backbone_model.load_state_dict(state_dict)
    backbone_model.classifier= classifier.simple_squential_classifier(in_features=backbone_model.last_channel, 
                                                                      n_classes=encoder_digit)

    return backbone_model