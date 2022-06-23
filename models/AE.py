import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ResNet


class BasicAE(nn.Module):

    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(  # 32x32x3
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 16x16x64
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 8x8x64
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 4x4x64
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
        #     nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
        #     nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
        # )
        # self.decoder2 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
        #     nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
        #     nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),
        # )
        self.decoder = nn.Sequential(                                       # 512 ,4 ,4
            nn.ConvTranspose2d(512, 64, kernel_size=2, stride=2, padding=0), #64 8 8
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0), # 64 16 16
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0),  # 3 32 32
        )

    def forward(self, x_input):
        # y = self.encoder(x_input)
        # bottel_neck = x.flatten()
        # bottel_neck = x.reshape()
        x = self.decoder(x_input)
        # z = self.decoder2(y)

        return x
