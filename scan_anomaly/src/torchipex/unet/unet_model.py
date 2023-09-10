#from .unet_parts import *
#from unet_parts import *
#from torch.utils.checkpoint import checkpoint
# importing torch from unet parts
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    # if time permits add frontend for binary classification=True
    def __init__(self, n_channels, 
                 n_classes, 
                 bilinear=False, 
                 binary=False, 
                 img_dim=64):
        super(UNet, self).__init__()
        self.binary = binary
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.img_dim = img_dim
        factor = 2 if bilinear else 1
        
        self.inc = (DoubleConv(n_channels, n_channels))
        #self.glob = (GlobalAvgPool())
        self.outc = (OutConv(n_channels, n_classes))
        self.outb = (FullyConected(in_channels=self.img_dim*img_dim))       

        # Calculate the number of down and up layers based on input dimensions
        num_layers = int(torch.floor(torch.log2(torch.tensor(float(img_dim)))))
        #print("Number of layers calculated: ", num_layers) 
        # Initialize down and up layers
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Create down layers
        in_channels=n_channels
        out_channels = in_channels
        for _ in range(num_layers - 2):
            #out_channels =  min(in_channels * 2, 512)
            out_channels =  in_channels * 2
            self.down_layers.append(Down(in_channels, out_channels))
            in_channels = out_channels

        in_channels=out_channels
        # Create up layers
        #i = 0 
        for i in range(num_layers - 3):
            out_channels = in_channels // 2
            self.up_layers.append(Up(in_channels,out_channels, bilinear))
            in_channels = out_channels
        
        # Add the last up layer
        self.up_last = (Up(out_channels, out_channels//2, bilinear))

    def forward(self, x):
        x0 = self.inc(x)
        # Down path
        down_outputs = []
        down_outputs.append(x0)
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            down_outputs.append(x)
            print()

        # Up path
        for idx, layer in enumerate(self.up_layers):
            x = layer(x, down_outputs[-(idx +2)])
            
        x = self.up_last(x, x0)
        
        logits = self.outc(x)
        
        if self.binary:
            logits = self.outb(logits)
        else:   
            logits = logits
        return logits
    
    def forward_debug(self, x):
        print("Shape of x: ", x.shape, "before inc")
        x0 = self.inc(x)
        print("Shape of x: ", x.shape, "after inc")
        # Down path
        down_outputs = []
        down_outputs.append(x0)
        for i, layer in enumerate(self.down_layers):
            print(f"Shape of x: {x.shape} before downlayer {i}")
            x = layer(x)
            print(f"Shape of x: {x.shape} after downlayer {i}")
            down_outputs.append(x)
            print()

        # Up path
        for idx, layer in enumerate(self.up_layers):
            print(f"Shape of x: {x.shape} before uplayer {idx}")
            print("x is fusing with the down layer of shape: ", down_outputs[-(idx +2)].shape) 
            x = layer(x, down_outputs[-(idx +2)])
            print(f"Shape of x: {x.shape} after uplayer {idx}")
            print()
            
        print("Shape of x: ", x.shape, "before up_last")
        x = self.up_last(x, x0)
        print("Shape of x: ", x.shape, "after up_last")
        
        logits = self.outc(x)
        print("Shape of logits: ", len(logits), logits[0].shape)
        
        if self.binary:
            logits = self.outb(logits)
        else:   
            logits = logits
        print("Checking n classes: ", self.n_classes)
        print("Shape of logits after if else: ", logits.shape)
        return logits
 

# for saving model in the main function
# def save_checkpoint(state, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)

# def train(net, dataloader, criterion, optimizer, device, num_epochs):
#     net.train()
#     net.to(device)
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             # Your training code here
#         # Save checkpoint at the end of the epoch
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'model_state_dict': net.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             # Add any other necessary state
#         })


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #print("x1 shape:", x1.shape)
        #print("x2 shape:", x2.shape)
        x1 = self.up(x1)
        #print("x1 shape after up:", x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #print("x shape after cat:", x.shape)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class FullyConected(nn.Module):
    """Flatten and add a Fully connected layer, use for binary classification task"""
    def __init__(self, in_channels=64*512, out_channels=1):
        super(FullyConected, self).__init__()
    
        self.fl = nn.Flatten()
        self.fc = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        #print("shape of input to flattend layer is ", x.shape)
        x = self.fl(x)
        #("shape of output of flattend layer output is ", x.shape)
        return self.fc(x)