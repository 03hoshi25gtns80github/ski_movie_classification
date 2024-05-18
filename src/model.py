import torch.nn as nn
from torchvision.models import resnet34
import timm

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        # 事前学習済みのResNet-34モデルをロード
        self.model = resnet34(pretrained=True)
        # 最後の全結合層を置き換え
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class CustomConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNeXt, self).__init__()
        # 事前学習済みのConvNeXtモデルをロード
        self.model = timm.create_model('convnext_base', pretrained=True)
        # 最後の全結合層を置き換え
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def create_model(num_classes):
    #return CustomResNet(num_classes=num_classes)
    return CustomConvNeXt(num_classes=num_classes)