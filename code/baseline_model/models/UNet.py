import torch
from torch import nn
import torchvision
from einops import rearrange


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class UNet(nn.Module):
    def __init__(self, n_classes, weight=None):
        super().__init__()

        self.base_model = torchvision.models.resnet50(True)
        if weight is not None:
            #! Need to change conv_layer
            self.base_model.load_state_dict(torch.load(weight))
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            self.base_layers[1],
            self.base_layers[2],
        )
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(2048, 1024 + 1024, 1024)
        self.decode3 = Decoder(1024, 512 + 512, 512)
        self.decode2 = Decoder(512, 256 + 256, 256)
        self.decode1 = Decoder(256, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
        )
        self.linear = nn.Linear(128*128, n_classes)

    def forward(self, input):
        # input [1,1,256,256]
        e1 = self.layer1(input)  # 64,128,128
        e2 = self.layer2(e1)  # 256,64,64
        e3 = self.layer3(e2)  # 512,32,32
        e4 = self.layer4(e3)  # 1024,16,16
        f = self.layer5(e4)  # 2048,8,8
        d4 = self.decode4(f, e4)  # 1024,16,16
        d3 = self.decode3(d4, e3)  # 512,32,32
        d2 = self.decode2(d3, e2)  # 256,64,64
        d1 = self.decode1(d2, e1)  # 64,128,128
        d0 = self.decode0(d1)  # 1,256,256
        out = torch.flatten(d0,1)
        out = self.linear(out)
        return out
