import torch
from torch import nn


class NonLocalAttention(nn.Module):
    """
        Attention from all positions in object B to all positions in object A
    """
    def __init__(self, in_channels=256, inter_channels=None, bn_layer=True):
        super(NonLocalAttention, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # Set default inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        bias=False, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, a, b):
        B, C, H, W = a.size()

        # Pairwise relationship
        theta_a = self.theta(a).reshape(B, self.inter_channels, -1).permute(0, 2, 1)
        phi_b = self.phi(b).reshape(B, self.inter_channels, -1)
        # Correlation of size (B, H * W, H * W)
        f = torch.matmul(theta_a, phi_b)
        f_div_C = f / f.size(-1)

        # Get representation of b
        g_b = self.g(b).view(B, self.inter_channels, -1)
        g_b = g_b.permute(0, 2, 1)

        # Combine
        y = torch.matmul(f_div_C, g_b)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_channels, *a.size()[2:])
        W_y = self.W(y)

        return W_y
