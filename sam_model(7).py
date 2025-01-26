import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class SAMSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(SAMSegmentation, self).__init__()
        self.encoder1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.middle = ConvBlock(512, 1024)

        self.decoder4 = ConvBlock(1024, 512)
        self.decoder3 = ConvBlock(512, 256)
        self.decoder2 = ConvBlock(256, 128)
        self.decoder1 = ConvBlock(128, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Middle
        m = self.middle(p4)

        # Decoder
        d4 = self.decoder4(F.interpolate(m, size=e4.size()[2:], mode='bilinear', align_corners=True))
        d3 = self.decoder3(F.interpolate(d4 + e4, size=e3.size()[2:], mode='bilinear', align_corners=True))
        d2 = self.decoder2(F.interpolate(d3 + e3, size=e2.size()[2:], mode='bilinear', align_corners=True))
        d1 = self.decoder1(F.interpolate(d2 + e2, size=e1.size()[2:], mode='bilinear', align_corners=True))

        # Classifier
        output = self.classifier(d1)
        return output