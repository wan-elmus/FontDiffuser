import torch
import torch.nn as nn
import torch.nn.functional as F

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, sn=True):
        super().__init__()
        self.sn = sn
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, bias=True
        )
        if self.sn:
            nn.utils.spectral_norm(self.conv)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BackgroundEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=48, max_channels=768, num_blocks=5, cross_attention_dim=1280):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.max_channels = max_channels
        self.num_blocks = num_blocks
        self.cross_attention_dim = cross_attention_dim

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        nn.utils.spectral_norm(self.conv_in)
        self.bn_in = nn.BatchNorm2d(base_channels)
        self.act_in = nn.LeakyReLU(0.2, inplace=True)

        # Downsampling blocks
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(num_blocks):
            next_channels = min(current_channels * 2, max_channels)
            self.blocks.append(
                DBlock(
                    in_channels=current_channels,
                    out_channels=next_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    sn=True
                )
            )
            current_channels = next_channels

        # Global average pooling and final linear layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(current_channels, cross_attention_dim)
        nn.utils.spectral_norm(self.fc)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.act_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Reshape for attention compatibility
        batch_size = x.size(0)
        x = x.view(batch_size, self.cross_attention_dim, 1, 1)
        return x