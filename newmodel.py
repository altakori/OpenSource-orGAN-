import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0, pooling=False):
        super().__init__()

        strides = 2
        if pooling:
            strides = 1
            
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        if pooling:
            layers.append(nn.MaxPool2d(2))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 32, normalize=False, pooling=False)
        self.down2 = UNetDown(32,64)                 
        self.down3 = UNetDown(64,128)               
        self.down4 = UNetDown(128,256) 
        self.down5 = UNetDown(256,256)      
        self.down6 = UNetDown(256,256)             
        self.down7 = UNetDown(256,256)              
        self.down8 = UNetDown(256,256,normalize=False)

        self.up1 = UNetUp(256,256,dropout=0.5)
        self.up2 = UNetUp(512,256,dropout=0.5)
        self.up3 = UNetUp(512,256,dropout=0.5)
        self.up4 = UNetUp(512,256)
        self.up5 = UNetUp(512,128)
        self.up6 = UNetUp(256,64)
        self.up7 = UNetUp(128,32)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(64,3,4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, strides=1, kernels_per_layer=1):
      super(depthwise_separable_conv, self).__init__()
      self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, stride=strides, padding=1, groups=nin)
      self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
      out = self.depthwise(x)
      out = self.pointwise(out)
      return out

class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, downsampling=True):
        super().__init__()

        strides = 1
        if downsampling:
            strides = 2
        
        layers = [depthwise_separable_conv(in_channels, out_channels, 3, strides=strides)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)#128
        self.stage_2 = Dis_block(64,128)#64
        self.stage_3 = Dis_block(128, 256)#32
        self.stage_4 = Dis_block(256,256, downsampling=False)#16
        self.patch = nn.Conv2d(256,1,3,padding=1) # 32x32 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x

def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)