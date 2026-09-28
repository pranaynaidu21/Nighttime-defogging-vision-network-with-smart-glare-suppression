import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(8, c // r)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, h, 1),
            nn.ReLU(True),
            nn.Conv2d(h, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class TransformerBlock(nn.Module):
    """Global context at 1/8 spatial resolution."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        q = self.n1(t)
        a, _ = self.attn(q, q, q, need_weights=False)
        t = t + a
        t = t + self.ffn(self.n2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class NightDehazeNet(nn.Module):
    """Laptop-sized CNN + channel attention + Transformer restoration network."""

    def __init__(self, base=24):
        super().__init__()
        b = base
        self.e1 = ConvBlock(3, b)
        self.d1 = nn.Conv2d(b, b * 2, 4, 2, 1)
        self.e2 = ConvBlock(b * 2, b * 2)
        self.d2 = nn.Conv2d(b * 2, b * 3, 4, 2, 1)
        self.e3 = ConvBlock(b * 3, b * 3)
        self.d3 = nn.Conv2d(b * 3, b * 4, 4, 2, 1)
        self.bottleneck = ConvBlock(b * 4, b * 4)
        self.attn = ChannelAttention(b * 4)
        self.tr = TransformerBlock(b * 4)

        self.u3 = nn.Sequential(nn.ConvTranspose2d(b * 4, b * 3, 4, 2, 1), nn.ReLU(True))
        self.f3 = ConvBlock(b * 6, b * 3)
        self.u2 = nn.Sequential(nn.ConvTranspose2d(b * 3, b * 2, 4, 2, 1), nn.ReLU(True))
        self.f2 = ConvBlock(b * 4, b * 2)
        self.u1 = nn.Sequential(nn.ConvTranspose2d(b * 2, b, 4, 2, 1), nn.ReLU(True))
        self.f1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        z = self.bottleneck(self.d3(e3))
        z = self.tr(self.attn(z))
        y = self.f3(torch.cat([self.u3(z), e3], dim=1))
        y = self.f2(torch.cat([self.u2(y), e2], dim=1))
        y = self.f1(torch.cat([self.u1(y), e1], dim=1))

        # Conservative residual: the network learns a correction rather than
        # replacing the input, reducing the chance of destroying clear areas.
        residual = 0.20 * torch.tanh(self.out(y))
        return torch.clamp(x + residual, 0, 1)
