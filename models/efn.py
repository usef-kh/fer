import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class EfficientNet(nn.Module):
    def __init__(self, pretrained=True, drop=0.1):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b3_ns', pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 7)

    def forward(self, x):
        # print(x.shape)
        x = x.repeat(1, 3, 1, 1)
        # print(x.shape)
        
        x = self.model(x)
        return x
