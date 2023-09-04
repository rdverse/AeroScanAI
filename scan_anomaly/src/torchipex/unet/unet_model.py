from .unet_parts import *
from unet_parts import *
from torch.utils.checkpoint import checkpoint
# importing torch from unet parts

class UNet(nn.Module):
    # if time permits add frontend for binary classification=True
    def __init__(self, n_channels, n_classes, bilinear=False, binary=False, imgdim=64):
        super(UNet, self).__init__()
        self.binary = binary
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.imgdim = imgdim
        factor = 2 if bilinear else 1
        
        self.inc = (DoubleConv(n_channels, 512))
        #self.glob = (GlobalAvgPool())
        self.outc = (OutConv(n_channels, n_classes))
        self.outb = (FullyConected(in_channels=self.imgdim*imgdim))       

        # Calculate the number of down and up layers based on input dimensions
        num_layers = int(torch.floor(torch.log2(torch.tensor(float(imgdim)))))
        print("Number of layers calculated: ", num_layers) 
        # Initialize down and up layers
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Create down layers
        in_channels=n_channels
        out_channels = in_channels
        print("in channels and out channels: ", in_channels, out_channels)
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
            
        print("Shape of logits after if else: ", logits)
        return logits
    
    
    def forward(self, x):
        x0 = self.inc(x)
        # Down path
        down_outputs = []
        down_outputs.append(x0)
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            down_outputs.append(x)

        # Up path
        for idx, layer in enumerate(self.up_layers):
            x = layer(x, down_outputs[-(idx +2)])
        #print("Shape of x: ", x.shape, "before up_last")
            
        x = self.up_last(x, x0)
        #print("Shape of x: ", x.shape, "after up_last")
        logits = self.outc(x)
        #print("Shape of logits: ", len(logits), logits[0].shape)
        #print("self.binary: ", self.binary)
        if self.binary:
         #   print("logits going to binary")
            logits = self.outb(logits)
        else:   
          #  print("logits going to softmax")
            logits = logits
        #print("Shape of logits after if else: ", logits)
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
