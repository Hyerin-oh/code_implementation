import torch
import torch.nn as nn

class Block(nn.Module):
  def __init__(self, in_channels ,out_channels ,repeat , bn_momentum):    # (3,64,2,0.1)
    super(Block, self).__init__()
    conv_in = nn.Conv2d(in_channels , out_channels , kernel_size = 3, padding = 1, stride = 1)
    conv_out = nn.Conv2d(out_channels , out_channels , kernel_size = 3, padding = 1, stride = 1)
    relu = nn.ReLU(True)
    bn = nn.BatchNorm2d(out_channels , momentum=bn_momentum)
    maxpool = nn.MaxPool2d(kernel_size= 2, stride = 2)

    layers = []
    for i in range(repeat):
      if (i == 0) : 
        layers.append(conv_in)
      else : 
        layers.append(conv_out)
      layers.append(bn)
      layers.append(relu)

    layers.append(maxpool)
    self.net = nn.Sequential(*layers)

  def forward(self,x):
    x = self.net(x)
    return 

class FCN(nn.Module):
    def __init__(self, num_classes, drop_r, bn_momentum , resolution):
        super(FCN , self).__init__()
        self.resolution = resolution

        # Backbone : VGG 19
        self.conv1 = self.make_block(3, 64 , 2, bn_momentum)
        self.conv2 = self.make_block(64, 128 , 2, bn_momentum)
        self.conv3 = self.make_block(128, 256 , 4, bn_momentum)
        self.conv4 = self.make_block(256, 512 , 4, bn_momentum)
        self.conv5 = self.make_block(512, 512 , 4, bn_momentum)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096 ,1)
        self.relu6 = nn.ReLU(True)
        self.drop6 = nn.Dropout2d(drop_r)

        # fc7
        self.fc7 = nn.Conv2d(4096 , 4096, 1)
        self.relu7 = nn.ReLU(True)
        self.drop7 = nn.Dropout2d(drop_r)

        # fc7
        self.score = nn.Conv2d(4096,num_classes,1)  #[1,7,7,num_classes]
        
        # fcn-32
        self.upsample32 = nn.ConvTranspose2d(num_classes , num_classes , kernel_size = 64 , stride = 32 , padding = 16)

        # fcn-16
        self.pool4_conv = nn.Conv2d(512 , num_classes , kernel_size = 1 , stride = 1 , padding = 0)
        self.upsample_double = nn.ConvTranspose2d(num_classes , num_classes , kernel_size = 4 , stride = 2 , padding = 1)
        self.upsample16 = nn.ConvTranspose2d(num_classes , num_classes , kernel_size = 32 , stride = 16 , padding = 8)

        #fcn-8
        self.pool3_conv = nn.Conv2d(256 , num_classes , kernel_size = 1 , stride = 1 , padding = 0)
        self.upsample8 = nn.ConvTranspose2d(num_classes , num_classes , kernel_size = 16 , stride = 8 , padding = 4)
         


    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      pool3 = self.pool3_conv(x)
      x = self.conv4(x)
      pool4 = self.pool4_conv(x)
      x = self.conv5(x)
      x = self.fc6(x)
      x = self.relu6(x)
      x = self.drop6(x)
      x = self.fc7(x)
      x = self.relu7(x)
      x = self.drop7(x)
      x = self.score(x)
      if (self.resolution == 32):
        x = self.upsample32(x)
        return x
      elif (self.resolution == 16):
        x = self.upsample_double(x) + pool4
        x = self.upsample16(x)
        return x
      else :
        x = self.upsample_double(x) + pool4
        x = self.upsample_double(x) + pool3
        x = self.upsample8(x)
        return x

    def make_block(self, in_channels ,out_channels ,repeat , bn_momentum):
      layers = []
      for i in range(repeat):
        if (i == 0) : 
          layers.append(nn.Conv2d(in_channels , out_channels , kernel_size = 3, padding = 1, stride = 1))
        else : 
          layers.append(nn.Conv2d(out_channels , out_channels , kernel_size = 3, padding = 1, stride = 1))
        layers.append(nn.BatchNorm2d(out_channels , momentum=bn_momentum))
        layers.append(nn.ReLU(True))

      layers.append(nn.MaxPool2d(kernel_size= 2, stride = 2))
      net = nn.Sequential(*layers)
      
      return net 

def FCN8(num_class):
    return FCN(num_class , 0.5 , 0.1 , 8) 

def FCN16(num_class ):
    return FCN(num_class , 0.5 , 0.1 , 16) 

def FCN32(num_class):
    return FCN(num_class , 0.5 , 0.1 , 32)
