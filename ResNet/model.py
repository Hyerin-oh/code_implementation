import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self ,out_channels , in_channels = None ,  downsample = False):
        super().__init__()
        
        self.downsample = downsample

        if self.downsample :
            in_channels = in_channels              # 64 -> 64 -> 256
            self.identity = nn.Sequential(nn.Conv2d(in_channels = in_channels , out_channels = out_channels ,kernel_size = 1, bias = True),
                            nn.BatchNorm2d(num_features = out_channels))
            self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = out_channels // 4,kernel_size = 1,bias = True)
        else :
            in_channels = out_channels                  # 256 -> 64 -> 256 
            self.conv1 = nn.Conv2d(in_channels = out_channels , out_channels = out_channels // 4,kernel_size = 1,bias = True)
        
        self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = out_channels // 4,kernel_size = 1,bias = True)
        self.conv2 = nn.Conv2d(in_channels = out_channels // 4 , out_channels = out_channels // 4, kernel_size = 3 ,stride = 1,padding = 1,bias = True )
        self.conv3 = nn.Conv2d(in_channels = out_channels // 4 , out_channels = out_channels , kernel_size = 1, bias = True )
        self.bn_in = nn.BatchNorm2d(num_features = out_channels // 4)
        self.bn_out = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU(True)

    def forward(self,x):
      
      if self.downsample :
        identity = self.identity(x)
      else :
        identity = x
      x = self.conv1(x)
      x = self.bn_in(x)
      x = self.relu(x)
      x = self.conv2 (x)
      x = self.bn_in(x)
      x = self.relu(x)
      x = self.conv3(x)
      x = self.bn_out(x)
      x = self.relu(x)
      out = x + identity
      out = self.relu(out)
      return out
        
        
class ResModel(nn.Module):
    def __init__(self, num_classes , block , repeat , out_channels):        # conv block이 stack되는 개수 , out_channels들의 list = [64, 256 ,512, 1024 ,2048 ]
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels = 3 ,out_channels = out_channels[0] , kernel_size= 7 , stride = 2 , padding = 3 , bias = True)
        self.bn = nn.BatchNorm2d(num_features = out_channels[0])
        self.relu = nn.ReLU(True)
        self.MaxPool = nn.MaxPool2d(kernel_size = 3 , stride = 2, padding = 1)
        self.Conv2 =  self.make_layer(block, repeat[0] , 64 , out_channels[1])
        self.Conv3 =  self.make_layer(block, repeat[1] ,out_channels[1] , out_channels[2])
        self.Conv4 =  self.make_layer(block, repeat[2] , out_channels[2] ,out_channels[3])
        self.Conv5 =  self.make_layer(block, repeat[3] ,out_channels[3] , out_channels[4])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channels[4] , num_classes)


    def forward(self, x):       # 3 x 224 x 224
        x = self.Conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.MaxPool(x)     # 64 x 112 x 112
        x = self.Conv2(x)       # 256 x 56 x 56
        x = self.Conv3(x)       # 512 x 28 x 28
        x = self.Conv4(x)       # 1024 x 14 x 14
        x = self.Conv5(x)       # 2048 x 7 x 7
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        return x   

  
    def make_layer(self, block , r , in_channel , out_channel):     # Residual block , for 문 횟수 , out_channel
        layers = []
        # 처음에는 차원을 맞추기 위한 projection 필요 64 -> 64 -> 256
        layers.append(block(out_channel , in_channel, True))
        # 그 이후에는 256 -> 64 -> 256
        for i in range(r-1):
          layers.append(block(out_channel))

        return nn.Sequential(*layers)


def ResNet50(num_class):
    return ResModel(num_class , ResidualBlock , [3,4,6,3] , [64, 256 ,512, 1024 ,2048] ) 
