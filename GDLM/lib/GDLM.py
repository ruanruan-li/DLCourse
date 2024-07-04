# torch libraries
import torch
import torch.nn as nn

# customized libraries
from .EfficientNet import EfficientNet
from .PVTv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4

# 定义卷积层、批归一化和ReLU组成的模块
class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

# 通道数减少模块
class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

# 软分组策略模块
class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)

# 梯度诱导转换模块
class GradientInducedTransition(nn.Module):
    def __init__(self, channel, M, N):
        super(GradientInducedTransition, self).__init__()
        self.M = M

        self.downsample2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)

        self.sgs3 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel + 32, channel, N=N)

    def forward(self, xr3, xr4, xr5, xg):
        # 传递梯度线索到上下文嵌入
        q3 = self.gradient_induced_feature_grouping(xr3, xg, M=self.M[0])
        q4 = self.gradient_induced_feature_grouping(xr4, self.downsample2(xg), M=self.M[1])
        q5 = self.gradient_induced_feature_grouping(xr5, self.downsample4(xg), M=self.M[2])

        # 注意力残差学习
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5

    def gradient_induced_feature_grouping(self, xr, xg, M):
        if M not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Invalid Group Number!: must be one of [1, 2, 4, 8, 16, 32]")

        if M == 1:
            return torch.cat((xr, xg), 1)

        xr_g = torch.chunk(xr, M, dim=1)
        xg_g = torch.chunk(xg, M, dim=1)
        foo = list()
        for i in range(M):
            foo.extend([xr_g[i], xg_g[i]])

        return torch.cat(foo, 1)

# 邻接连接解码器模块
class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3):
        zt5_1 = zt5
        zt4_1 = self.conv_upsample1(self.upsample(zt5)) * zt4
        zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) * zt3

        zt4_2 = torch.cat((zt4_1, self.conv_upsample4(self.upsample(zt5_1))), 1)
        zt4_2 = self.conv_concat2(zt4_2)

        zt3_2 = torch.cat((zt3_1, self.conv_upsample5(self.upsample(zt4_2))), 1)
        zt3_2 = self.conv_concat3(zt3_2)

        pc = self.conv4(zt3_2)
        pc = self.conv5(pc)

        return pc

# 纹理编码器模块
class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.conv1 = ConvBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBR(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBR(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        xg = self.conv3(feat)
        pg = self.conv_out(xg)
        return xg, pg

# DGNet主网络结构
class DGNet(nn.Module):
    def __init__(self, channel=32, arc='B0', M=[8, 8, 8], N=[4, 8, 16]):
        super(DGNet, self).__init__()

        # 根据选择的架构初始化上下文编码器
        if arc.startswith('EfficientNet'):
            print('--> using {} right now'.format(arc))
            self.context_encoder = EfficientNet.from_pretrained(arc.lower())
            in_channel_list = {
                'efficientnet-b1': [40, 112, 320],
                'efficientnet-b4': [56, 160, 448]
            }[arc.lower()]
        elif arc.startswith('PVTv2'):
            print('--> using {} right now'.format(arc))
            self.context_encoder = globals()[arc.lower()](pretrained=True)
            in_channel_list = {
                'pvt_v2_b0': [64, 160, 256],
                'pvt_v2_b1': [128, 320, 512],
                'pvt_v2_b2': [128, 320, 512],
                'pvt_v2_b3': [128, 320, 512],
                'pvt_v2_b4': [128, 320, 512]
            }[arc.lower()]
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))

        self.texture_encoder = TextureEncoder()

        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=channel)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=channel)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=channel)

        self.git = GradientInducedTransition(channel=channel, M=M, N=N)
        self.ncd = NeighborConnectionDecoder(channel=channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 上下文路径（编码器）
        endpoints = self.context_encoder.extract_endpoints(x)
        x3 = endpoints['reduction_3']
        x4 = endpoints['reduction_4']
        x5 = endpoints['reduction_5']

        xr3 = self.dr3(x3)
        xr4 = self.dr4(x4)
        xr5 = self.dr5(x5)

        # 空间路径（编码器）
        xg, pg = self.texture_encoder(x)

        # 解码器
        zt3, zt4, zt5 = self.git(xr3, xr4, xr5, xg)
        pc = self.ncd(zt5, zt4, zt3)

        return self.upsample(pc), self.upsample(pg)


if __name__ == '__main__':
    net = DGNet(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).eval()
    inputs = torch.randn(1, 3, 352, 352)
    outs = net(inputs)
    print(outs[0].shape)
