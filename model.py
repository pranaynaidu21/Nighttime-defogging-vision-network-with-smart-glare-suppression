import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(4, c // r)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, h, 1), nn.ReLU(True),
            nn.Conv2d(h, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.a = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.f = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        q = self.n1(t)
        a, _ = self.a(q, q, q, need_weights=False)
        t = t + a
        t = t + self.f(self.n2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)

class NightDehazeNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(True)
        )
        self.d = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.e2 = nn.Sequential(
            nn.ReLU(True), nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU(True)
        )
        self.attn = ChannelAttention(base * 2)
        self.tr = TransformerBlock(base * 2)
        self.u = nn.Sequential(nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(True))
        self.f = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(base, 3, 3, padding=1)
    def forward(self, x):
        e = self.e1(x)
        z = self.tr(self.attn(self.e2(self.d(e))))
        y = self.u(z)
        y = self.f(torch.cat([y, e], 1))
        return torch.clamp(x + 0.5 * torch.tanh(self.out(y)), 0, 1)
