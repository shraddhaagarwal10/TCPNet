from collections import OrderedDict
import os
import torch
import torch.nn as nn

class TCPNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(TCPNet, self).__init__()
        features = init_features

        self.encoder1 = TCPNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_drop1 = nn.Dropout(0.1)
        self.encoder2 = TCPNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_drop2 = nn.Dropout(0.1)
        self.encoder3 = TCPNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_drop3 = nn.Dropout(0.1)
        self.encoder4 = TCPNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_drop4 = nn.Dropout(0.1)

        self.bottleneck = TCPNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = TCPNet._block((features * 8) * 2, features * 8, name="dec4")
        self.dec_drop4 = nn.Dropout(0.1)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = TCPNet._block((features * 4) * 2, features * 4, name="dec3")
        self.dec_drop3 = nn.Dropout(0.1)

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = TCPNet._block((features * 2) * 2, features * 2, name="dec2")
        self.dec_drop2 = nn.Dropout(0.1)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = TCPNet._block(features * 2, features, name="dec1")
        self.dec_drop1 = nn.Dropout(0.1)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.conv_var = nn.Conv2d(
            in_channels=features, out_channels=1, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        drop_enc1 = self.enc_drop1(enc1)
        enc2 = self.encoder2(self.pool1(drop_enc1))
        drop_enc2 = self.enc_drop2(enc2)
        enc3 = self.encoder3(self.pool2(drop_enc2))
        drop_enc3 = self.enc_drop3(enc3)
        enc4 = self.encoder4(self.pool3(drop_enc3))
        drop_enc4 = self.enc_drop4(enc4)
        bottleneck = self.bottleneck(self.pool4(drop_enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        drop_dec4 = self.dec_drop4(dec4)
        dec3 = self.upconv3(drop_dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        drop_dec3 = self.dec_drop3(dec3)
        dec2 = self.upconv2(drop_dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        drop_dec2 = self.dec_drop2(dec2)
        dec1 = self.upconv1(drop_dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        pred_dec1 = self.decoder1(dec1)
        pred = self.dec_drop1(pred_dec1)
        var_dec1 = self.decoder1(dec1)
        var = self.dec_drop1(var_dec1)
        return torch.sigmoid(self.conv(pred)), torch.sigmoid(self.conv_var(var))

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
im = torch.randn(32,1,64,64)
mod = TCPNet(out_channels=2)
mod(im)
