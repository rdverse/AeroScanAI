import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F

INPUT_IMG_SIZE = (64, 64)
# Define a custom CNN module to convert 10 channels to 3
class CustomCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CustomCNN2(nn.Module):
    def __init__(self, in_channels, hidden_channels=1, out_channels=1):
        super(CustomCNN2, self).__init__()
        hidden_channels = in_channels // 2
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, dilation=2)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, dilation=10)
    def forward(self, x):
        return self.conv2(self.conv(x))

# Define your PyTorch model (replace this with your actual model)
class CustomSqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomSqueezeNet, self).__init__()
        # Add the custom CNN module to convert 10 channels to 3
        self.cnn = CustomCNN(10, 3)
        self.feature_extractor = models.squeezenet1_0(pretrained=True).features[:-1]

        print(self.feature_extractor[-1])
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(INPUT_IMG_SIZE[0] // 2 ** 5, INPUT_IMG_SIZE[1] // 2 ** 5)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=256,#self.feature_extractor[-1].out_channels,
                out_features=num_classes,
            ),
        )

    def _freeze_params(self):
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False

    def forward(self, x_in):
        """
        forward
        """
        # Apply the custom CNN module to convert 10 channels to 3
        x_in = self.cnn(x_in)
        feature_maps = self.feature_extractor(x_in)
        print(feature_maps.shape)
        scores = self.classification_head(feature_maps)
        if self.training:
            return scores
        return