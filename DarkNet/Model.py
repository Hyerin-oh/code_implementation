class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlock ,self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.conv1x1 = nn.Conv2d(in_channels = self.in_channels,
                              out_channels = self.out_channels,
                               kernel_size = 1,
                               stride = 1,
                               bias = True
                              ) # 64d -> 1x1 32d

        self.conv3x3 = nn.Conv2d(in_channels = self.out_channels,
                              out_channels = self.in_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = True
                              ) # 32d -> 1x1, 64d
        self.bn1x1 = nn.BatchNorm2d(num_features = self.out_channels)
        self.bn3x3 = nn.BatchNorm2d(num_features =self.in_channels)
        self.relu = nn.LeakyReLU(True)
        
    def forward(self,x):
        base = x 
        x = self.conv1x1(x)
        x = self.bn1x1(x)
        x = self.relu(x)
        x = self.conv3x3(x)
        x = self.bn3x3(x)
        x = self.relu(x)
        return x + base 
        
        
class ConvS2(nn.Module):
    def __init__(self, in_channels , out_channels):
        super(ConvS2 , self).__init__()
        self.conv= nn.Conv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1,
                                bias = True)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.LeakyReLU(True)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

        
class DarkModel(nn.Module):
    def __init__(self, num_class , ResBlock , ConvBlock):
        super(DarkModel,self).__init__()
        self.num_class = num_class

        self.conv0 = nn.Conv2d(in_channels = 3 , 
                                out_channels = 32,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                bias = True)
        self.bn = nn.BatchNorm2d(num_features = 32)
        self.conv1 = ConvBlock(in_channels = 32 , out_channels =64)
        self.ResBlock1 = self.make_layer(ResBlock(in_channels = 64), 1)
        self.conv2 = ConvBlock(in_channels = 64 , out_channels = 128)  
        self.ResBlock2 = self.make_layer(ResBlock(in_channels = 128), 2)
        self.conv3 = ConvBlock(in_channels = 128 , out_channels = 256)
        self.ResBlock3 = self.make_layer(ResBlock(in_channels = 256), 8)
        self.conv4 = ConvBlock(in_channels = 256 , out_channels  =512)
        self.ResBlock4 =self.make_layer(ResBlock(in_channels = 512), 8)
        self.conv5 = ConvBlock(in_channels = 512 , out_channels = 1024)
        self.ResBlock5 = self.make_layer(ResBlock(in_channels = 1024), 4)
        self.relu = nn.LeakyReLU(True)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, self.num_class)



    def forward(self,x):
        x = self.conv0(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.ResBlock1(x)
        x = self.conv2(x)
        x = self.ResBlock2(x)
        x = self.conv3(x)
        x = self.ResBlock3(x)
        x = self.conv4(x)
        x = self.ResBlock4(x)
        x = self.conv5(x)
        x = self.ResBlock5(x)
        x = self.GAP(x)
        x = x.squeeze()
        x = self.fc(x)
        return x         

    
    
    def make_layer(self,block,repeat):
        layers = []
        for i in range(repeat):
            layers.append(block)
        return nn.Sequential(*layers)
    

def DarkNet53(num_class):
    return DarkModel(num_class , ResidualBlock , ConvS2) 
