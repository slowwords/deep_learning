import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module): # 基础的卷积块，结构为Conv-BN-ReLU
    def __init__(self, in_dim, out_dim, down=True):
        super(ConvLayer, self).__init__()
        if down:
            stride = 2
        else:
            stride = 1
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class DecoderLayer(nn.Module):
    def __init__(self, in_dim, out_dim, down=False):
        super(DecoderLayer, self).__init__()
        self.conv = ConvLayer(in_dim, out_dim, down=down)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")   # decoder使用双线性插值进行上采样
        return self.conv(x)

class AE(nn.Module):
    def __init__(self, opts):
        super(AE, self).__init__()
        in_dim = opts.in_dims
        out_dim = opts.latent_dims
        self.E = nn.Sequential(
            ConvLayer(in_dim, out_dim//4),
            ConvLayer(out_dim//4, out_dim//2),
            ConvLayer(out_dim//2, out_dim)
        )
        self.D = nn.Sequential(
            DecoderLayer(out_dim, out_dim//2),
            DecoderLayer(out_dim//2, out_dim//4),
            DecoderLayer(out_dim//4, in_dim)
        )
    def forward(self, image, return_latent=False):
        latent = self.E(image)
        out = self.D(latent)
        if return_latent:
            return out, latent
        else:
            return out

if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)
    net = AE(3, 128)
    y, l = net(x, return_latent=True)
    from IPython import embed
    embed()