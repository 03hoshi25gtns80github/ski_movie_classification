import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, base_model, out_dim):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.fc.in_features, out_dim)
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

def get_model(out_dim):
    base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # weights引数を使用
    model = Encoder(base_model, out_dim)
    return model
