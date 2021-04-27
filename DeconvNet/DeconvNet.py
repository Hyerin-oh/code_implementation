'''
reference 
http://cvlab.postech.ac.kr/research/deconvnet/model/DeconvNet/DeconvNet_inference_deploy.prototxt
'''
import torch
import torch.nn as nn

class DeconvNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DeconvNet, self).__init__()
        self.dropout = nn.Dropout2d()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True) # unpooling을 위함
        self.unpool = nn.MaxUnpool2d(kernel_size = 2, stride = 2) # unpooling을 위함
        self.conv1 = self.make_CB(2,3,64)
        self.conv2 = self.make_CB(2,64,128)
        self.conv3 = self.make_CB(3,128,256)
        self.conv4 = self.make_CB(3,256,512)
        self.conv5 = self.make_CB(3,512,512)
        self.conv_7x7 = nn.Conv2d(512, 4096, 7)
        self.conv_1x1 = nn.Conv2d(4096, 4096, 1)
        self.deconv_7x7 = nn.ConvTranspose2d(4096, 512, 7)
        self.deconv5 = self.make_DB(3,512,512)
        self.deconv4 = self.make_DB(3,512,256)
        self.deconv3 = self.make_DB(3,256,128)
        self.deconv2 = self.make_DB(2,128,64)
        self.deconv1 = self.make_DB(2,64,64)
        self.score = nn.Conv2d(64,num_classes,1)

    def forward(self, x):
        x = self.conv1(x)
        x , pool1 = self.pool(x)
        x = self.conv2(x)
        x , pool2 = self.pool(x)
        x = self.conv3(x)
        x , pool3 = self.pool(x)
        x = self.conv4(x)
        x , pool4 = self.pool(x)
        x = self.conv5(x)
        x , pool5 = self.pool(x)
        x = self.conv_7x7(x)
        x = self.dropout(x)
        x = self.conv_1x1(x)
        x = self.dropout(x)
        x = self.deconv_7x7(x)
        x = self.unpool(x ,pool5)
        x = self.deconv5(x)
        x = self.unpool(x ,pool4)
        x = self.deconv4(x)
        x = self.unpool(x ,pool3)
        x = self.deconv3(x)
        x = self.unpool(x ,pool2)
        x = self.deconv2(x)
        x = self.unpool(x ,pool1)
        x = self.deconv1(x)
        x = self.score(x)

        return x
    
    def make_DB(self, repeat, in_channels , out_channels, kernel_size = 3 , stride = 1 , padding = 1):
        layers = []
        for i in range(repeat):
            if (i==0):
                layers.append(nn.ConvTranspose2d(in_channels, out_channels,kernel_size, stride = stride , padding = padding))
            else :
                layers.append(nn.ConvTranspose2d(out_channels, out_channels,kernel_size, stride = stride , padding = padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)
    
    def make_CB(self, repeat, in_channels , out_channels, kernel_size = 3 , stride = 1 , padding = 1):
        layers = []
        for i in range(repeat):
            if (i == 0):
                layers.append(nn.Conv2d(in_channels, out_channels,kernel_size, stride = stride , padding = padding))
            else :
                layers.append(nn.Conv2d(out_channels, out_channels,kernel_size, stride = stride , padding = padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))

        return nn.Sequential(*layers)
