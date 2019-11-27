from collections import OrderedDict

import torch
import torch.nn as nn
from models.layers.grid_attention_layer import GridAttentionBlock2D
from models.networks.utils import UnetGridGatingSignal2
class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, nonlocal_mode='concatenation', attention_dsample=(2,2)):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        self.gating = UnetGridGatingSignal2(features * 16, features * 8, kernel_size=(1, 1))

        # attention blocks
        self.attentionblock2 = GridAttentionBlock2D(in_channels=features * 2, gating_channels=features * 8,
                                                    inter_channels=features * 2, sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock2D(in_channels=features * 4, gating_channels=features * 8,
                                                    inter_channels=features * 4, sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock2D(in_channels=features * 8, gating_channels=features * 8,
                                                    inter_channels=features * 8, sub_sample_factor=attention_dsample, mode=nonlocal_mode)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        gating = self.gating(bottleneck)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(enc4, gating)
        g_conv3, att3 = self.attentionblock3(enc3, gating)
        g_conv2, att2 = self.attentionblock2(enc2, gating)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((g_conv4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((g_conv3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((g_conv2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
