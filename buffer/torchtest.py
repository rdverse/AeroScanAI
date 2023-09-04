import torch
from torch import nn
from torchvision.models import squeezenet1_0

from squeezenet import *

# class CustomSqueezeNet(nn.Module):
#     def __init__(self, n_classes=2):
#         super().__init__()
#         self.feature_extractor = squeezenet1_0(pretrained=True).features
#         print(self.feature_extractor[-1])
        
#         self.segmentation_head = nn.Conv2d(
#             in_channels=self.feature_extractor[-1].out_channels,
#             out_channels=n_classes,
#             kernel_size=1,
#         )

#     def forward(self, x_in):
#         """
#         Forward pass
#         """
#         feature_maps = self.feature_extractor(x_in)
#         segmentation_map = self.segmentation_head(feature_maps)

#         return segmentation_map


# class CustomSqueezeNet2(nn.Module):
#     def __init__(self, n_classes=2):
#         super(CustomSqueezeNet2, self).__init__()
        
#         # Load the pre-trained SqueezeNet model
#         squeezenet = models.squeezenet1_0(pretrained=True)
        
#         # Extract the feature extractor layers
#         self.feature_extractor = squeezenet.features
        
#         # Define your custom layers for classification
#         self.classification_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
#             nn.Flatten(),
#             nn.Linear(512, n_classes)  # Adjust the input size based on your needs
#         )

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         output = self.classification_head(features)
#         return output




num_output_channels = 256


import torch

# Define the input shape and batch size
batch_size = 4  # You can change this to your desired batch size
num_channels = 10
height, width = 64, 64

# Generate random values for the input tensor
input_tensor = torch.randn(batch_size, num_channels, height, width)

# Print the shape of the generated input tensor
print("Shape of input tensor:", input_tensor.shape)

# Instantiate the custom model
model = CustomSqueezeNet(num_classes=2)

# Forward pass to get segmentation predictions
segmentation_predictions = model(input_tensor)  # input_tensor should be (batch_size, 10, 64, 64)

# You can apply softmax or any other post-processing as needed for segmentation
segmentation_probabilities = F.softmax(segmentation_predictions, dim=1)