""" Full assembly of the parts to form the complete network """
import torch.nn as nn

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = (DoubleConv(n_channels, 64))
        # self.down1 = (Down(64, 128))
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512))
        # self.down4 = (Down(512, 1024))
        # self.down5 = (Down(1024, 2048))
        # factor = 2 if bilinear else 1
        # self.down6 = (Down(2048, 4096//factor))
        # self.up1 = (Up(4096, 2048 // factor, bilinear))
        # self.up2 = (Up(2048, 1024 // factor, bilinear))
        # self.up3 = (Up(1024, 512 // factor, bilinear))
        # self.up4 = (Up(512, 256 // factor, bilinear))
        # self.up5 = (Up(256, 128 // factor, bilinear))
        # self.up6 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(64, n_classes))
        factor = 2 if bilinear else 1  # ✅ define before use

        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1   = Up(1024, 512 // factor, bilinear)
        self.up2   = Up(512, 256 // factor, bilinear)
        self.up3   = Up(256, 128 // factor, bilinear)
        self.up4   = Up(128, 64, bilinear)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x6 = self.down5(x5)
        # x7 = self.down6(x6)
        # x = self.up1(x7,x6)
        # x = self.up2(x, x5)
        # x = self.up3(x, x4)
        # x = self.up4(x, x3)
        # x = self.up5(x, x2)
        # x = self.up6(x, x1)
        # logits = self.outc(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        # self.inc = torch.utils.checkpoint(self.inc)
        # self.down1 = torch.utils.checkpoint(self.down1)
        # self.down2 = torch.utils.checkpoint(self.down2)
        # self.down3 = torch.utils.checkpoint(self.down3)
        # self.down4 = torch.utils.checkpoint(self.down4)
        # self.down5 = torch.utils.checkpoint(self.down5)
        # self.down6 = torch.utils.checkpoint(self.down6)
        # self.up1 = torch.utils.checkpoint(self.up1)
        # self.up2 = torch.utils.checkpoint(self.up2)
        # self.up3 = torch.utils.checkpoint(self.up3)
        # self.up4 = torch.utils.checkpoint(self.up4)
        # self.up5 = torch.utils.checkpoint(self.up5)
        # self.up6 = torch.utils.checkpoint(self.up6)
        # self.outc = torch.utils.checkpoint(self.outc)
        self.inc = torch.utils.checkpoint.checkpoint_sequential([self.inc], 1)
        self.down1 = torch.utils.checkpoint.checkpoint_sequential([self.down1], 1)
        self.down2 = torch.utils.checkpoint.checkpoint_sequential([self.down2], 1)
        self.down3 = torch.utils.checkpoint.checkpoint_sequential([self.down3], 1)
        self.down4 = torch.utils.checkpoint.checkpoint_sequential([self.down4], 1)
        self.up1   = torch.utils.checkpoint.checkpoint_sequential([self.up1], 1)
        self.up2   = torch.utils.checkpoint.checkpoint_sequential([self.up2], 1)
        self.up3   = torch.utils.checkpoint.checkpoint_sequential([self.up3], 1)
        self.up4   = torch.utils.checkpoint.checkpoint_sequential([self.up4], 1)
        self.outc  = torch.utils.checkpoint.checkpoint_sequential([self.outc], 1)

