import numbers
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from einops import rearrange
from fusion_strategy import *
from utils import *


class Convlutionlayer(nn.Module):
    """
        self-build Convolution series
    """
    def __init__(self, in_c, out_c, k, s, pad=True):
        super(Convlutionlayer, self).__init__()
        self.pad = pad
        self.ref_pad = nn.ReflectionPad2d(k // 2)
        self.conv = nn.Conv2d(in_c, out_c, k, s, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.activate = nn.LeakyReLU(out_c)

    def forward(self, x):
        if self.pad:
            return self.activate(self.bn(self.conv(self.ref_pad(x))))
        else:
            return self.activate(self.bn(self.conv(x)))


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_fratures,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.module_in = nn.Conv2d(
            in_features, hidden_features * 4, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 4, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias, padding_mode="reflect")
        self.norm = nn.BatchNorm2d(hidden_features)

        self.module_out = nn.Conv2d(
            hidden_features, out_fratures, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.module_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.norm(x)
        x = self.module_out(x)
        return x


class FReLU(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv2d = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c)
        self.bn = nn.BatchNorm2d(in_c)

    def forward(self, x):
        x1 = self.bn(self.conv2d(x))
        x2 = torch.stack([x, x1], dim=0)
        out, _ = torch.max(x2, dim=0)
        return out


class DWConv(nn.Module):
    """
        深度可分离卷积DWConv的标准实现代码
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DWConv, self).__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_c, stride=1):
        super().__init__()
        self.r = 2
        self.conv = nn.Conv2d(in_c, in_c * self.r, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(in_c * self.r)

        self.conv1 = nn.Conv2d(in_c, 2 * in_c * self.r, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * in_c * self.r)
        self.prelu1 = FReLU(2 * in_c * self.r)

        self.conv2 = nn.Conv2d(2 * in_c * self.r, 2 * in_c * self.r, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(2 * in_c * self.r)
        self.prelu2 = FReLU(2 * in_c * self.r)

        self.conv3 = nn.Conv2d(2 * in_c * self.r, in_c * self.r, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_c * self.r)
        self.prelu3 = FReLU(in_c * self.r)

    def forward(self, x):
        identity = self.conv(x)
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.prelu3(self.bn3(self.conv3(out)))
        out = out + identity
        return self.relu(out)


class MTEFuse_Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(MTEFuse_Encoder, self).__init__()
        self.inplanes = 16
        self.pool = nn.MaxPool2d(2)
        # 多尺度特征层特征提取
        self.base_feature = nn.Sequential(
            nn.Conv2d(in_c, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(16),)
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2)
        self.layer3 = self._make_layer(out_c, 2)

    def _make_layer(self, channels, stride=1):
        layers = []
        layers.append(Bottleneck(channels, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # [16, 32, 64, 128]
        x1 = self.base_feature(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        return [x1, x2, x3, x4]


def CMDAF(F_vi, F_ir):
    # 计算差分
    sub_vi_ir = F.relu(F_vi - F_ir)
    sub_ir_vi = F.relu(F_ir - F_vi)

    # 全局平均池化 + Sigmoid生成权重
    sub_w_vi_ir = torch.mean(sub_vi_ir, dim=[2, 3], keepdim=True)  # B x C x 1 x 1
    w_vi_ir = torch.sigmoid(sub_w_vi_ir)

    sub_w_ir_vi = torch.mean(sub_ir_vi, dim=[2, 3], keepdim=True)
    w_ir_vi = torch.sigmoid(sub_w_ir_vi)

    # 差分增强
    F_dvi = w_vi_ir * sub_ir_vi *50
    F_dir = w_ir_vi * sub_vi_ir *50

    # 特征融合
    F_fvi = F_vi + F_dir
    F_fir = F_ir + F_dvi

    return F_fvi, F_fir

class DFA(nn.Module):
    def __init__(self, out_dim):
        super(DFA, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_dim * 2)
        self.vi_attn=nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.ir_attn=nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        vi, ir = x.chunk(2, dim=1)
        vi_enhance = vi*(1+self.ir_attn(vi))
        ir_enhance = ir*(1+self.vi_attn(ir))
        sub_vi_ir = F.relu(vi_enhance - ir_enhance)
        sub_ir_vi = F.relu(ir_enhance - vi_enhance)

        # 全局平均池化 + Sigmoid生成权重
        sub_w_vi_ir = torch.mean(sub_vi_ir, dim=[2, 3], keepdim=True)  # B x C x 1 x 1
        w_vi_ir = torch.sigmoid(sub_w_vi_ir)

        sub_w_ir_vi = torch.mean(sub_vi_ir, dim=[2, 3], keepdim=True)
        w_ir_vi = torch.sigmoid(sub_w_ir_vi)

        # 局部自适应缩放
        scale_factor_vi = self.sigmoid(self.conv((F.adaptive_avg_pool2d(w_ir_vi, (1, 1))).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)).expand_as(w_ir_vi)
        scale_factor_ir = self.sigmoid(self.conv((F.adaptive_avg_pool2d(w_vi_ir, (1, 1))).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)).expand_as(w_vi_ir)
        F_dvi = sub_vi_ir * scale_factor_vi
        F_dir = sub_ir_vi * scale_factor_ir

        # 特征融合
        F_fvi = ir_enhance + F_dvi
        F_fir = vi_enhance + F_dir

        F_shallow = torch.concat((F_fvi, F_fir), dim=1)
        return F_shallow

class TRCNB(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TRCNB, self).__init__()
        self.embed = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.GlobalFeature = GlobalFeatureExtractor(dim=out_dim, num_heads=8)
        # self.LocalFeature = LocalFeatureExtractor(dim=out_dim, num_blocks=3, growth_rate=32)
        self.LocalFeature = LocalFeatureExtractor(dim=out_dim)
        self.FFN = nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1, bias=False,
                             padding_mode="reflect")
    def forward(self, x):
        x = self.embed(x)
        x1 = self.GlobalFeature(x)+x
        x2 = self.LocalFeature(x)+x
        x3 = torch.cat((x1,x2),1)
        out = self.FFN(x3)+x
        return out

class MTEFuse_Infuse(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MTEFuse_Infuse, self).__init__()
        self.dfa=DFA(out_dim)
        self.trcnb=TRCNB(in_dim, out_dim)
    def forward(self, x):
        x=self.dfa(x)
        out=self.trcnb(x)
        return out

class GlobalFeatureExtractor(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(GlobalFeatureExtractor, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Transformer_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(in_features=dim, out_fratures=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)  # 这个是你自定义的带偏置 LayerNorm

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 转为 [B, H*W, C]
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # LayerNorm 操作
        x_normed = self.body(x_reshaped)
        # 再变回 [B, C, H, W]
        x_out = x_normed.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_out


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class Transformer_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.5,
            proj_drop: float = 0.5,
            use_flash_attn: bool = False
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 固定缩放因子 (标准注意力实现)
        self.use_flash_attn = use_flash_attn

        # 分离的 Q/K/V 卷积投影
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        # 深度可分离卷积增强局部特征提取
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=qkv_bias)

        # 投影层和正则化
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for conv in [self.q_conv, self.k_conv, self.v_conv]:
            xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        # nn.init.zeros_(self.proj.bias)

    def _reshape(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """将特征图转换为多头注意力需要的形状"""
        B, C = x.shape[:2]
        return x.view(B, self.num_heads, self.head_dim, H * W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 局部特征增强
        x = self.dw_conv(x)

        # 独立的 Q/K/V 投影
        q = self.q_conv(x)  # [B, C, H, W]
        k = self.k_conv(x)
        v = self.v_conv(x)

        # 转换为多头表示
        q = torch.nn.functional.normalize(self._reshape(q, H, W), dim=-1)  # [B, num_heads, head_dim, (H*W)]
        k = torch.nn.functional.normalize(self._reshape(k, H, W), dim=-1)
        v = torch.nn.functional.normalize(self._reshape(v, H, W), dim=-1)
        # qkv运算
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, num_heads, head_dim, head_dim]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v) # [B, num_heads, head_dim, (H*W)]
        # 合并多头
        out = out.contiguous().view(B, C, H, W) # [B, (num_heads*head_dim), H, W]
        # 特征投影层及正则化
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

########################################################################################################################################

def find_largest_factor(n, max_factor):
    for f in reversed(range(1, max_factor + 1)):
        if n % f == 0:
            return f
    return 1  # fallback to 1 group

class Channel_Attention(nn.Module):
    def __init__(self, channels, factor=32):
        super(Channel_Attention, self).__init__()
        self.groups = find_largest_factor(channels, factor)
        assert channels % self.groups == 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(1, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # 增加 skip connection 避免信息被压光
        enhanced = group_x * weights.sigmoid()
        return (0.8 * enhanced + 0.2 * group_x).reshape(b, c, h, w)


class CHSP(nn.Module):
    def __init__(self, channels, factor=32, reduction=4):
        super().__init__()
        self.channel = Channel_Attention(channels, factor=factor)
        self.spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel = self.channel(x)
        spatial_weight = self.spatial(x)
        spatial = x * (0.4 + 0.6 * spatial_weight)
        return channel + spatial


class DenseResBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, 3, 1, 1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 1, 1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.GroupNorm(1, in_channels + 2 * growth_rate)  # normalize dense features
        self.chsp = CHSP(in_channels + 2 * growth_rate)
        self.fusion = nn.Conv2d(in_channels + 2 * growth_rate, in_channels, 1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(torch.cat([x, out1], dim=1))
        dense = torch.cat([x, out1, out2], dim=1)
        # texture = self.texture_branch(x)  # 保留高频结构
        chsp_attention = self.chsp(dense)
        fused = self.fusion(chsp_attention)
        return fused + x


class LocalFeatureExtractor(nn.Module):
    def __init__(self, dim=64, num_blocks=3, growth_rate=32):
        super(LocalFeatureExtractor, self).__init__()
        self.blocks = nn.Sequential(
            *[DenseResBlock(dim, growth_rate) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)

########################################################################################################################################

class DecoderConvlution(nn.Module):
    def __init__(self, in_c, out_c, k, s, pad=True, activation='prelu'):
        super(DecoderConvlution, self).__init__()
        self.pad = pad
        self.need_proj = (in_c != out_c)
        self.ref_pad = nn.ReflectionPad2d(k // 2) if pad else nn.Identity()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c//2, k, s),
                                  nn.BatchNorm2d(out_c//2),
                                  nn.PReLU(),
                                  nn.Conv2d(out_c // 2, out_c, s),
                                  nn.BatchNorm2d(out_c))

        self.proj = nn.Conv2d(in_c, out_c, 1, 1, bias=False) if self.need_proj else nn.Identity()

        if activation == 'leakyrelu':
            self.activate = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.activate = nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        identity = self.proj(x)
        out = self.conv(self.ref_pad(x))
        out = self.activate(out)
        return out + identity


class HighFreqBranch(nn.Module):
    def __init__(self, channels):
        super(HighFreqBranch, self).__init__()
        # Laplacian 卷积核（每通道分别处理）
        self.laplace = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            bias=False, groups=channels  # 每通道单独卷积
        )
        lap_kernel = torch.tensor([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]], dtype=torch.float32)
        lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.laplace.weight.data = lap_kernel
        self.laplace.weight.requires_grad = False  # 不训练，固定边缘提取

    def forward(self, x):
        high_freq = self.laplace(x)
        return x + high_freq  # 残差增强


class MTEFuse_Decoder(nn.Module):
    def __init__(self):
        super(MTEFuse_Decoder, self).__init__()
        # 上采样模块
        self.up_eval2 = UpReshape(2)
        self.up_eval4 = UpReshape(4)
        self.up_eval8 = UpReshape(8)

        # 下采样模块
        self.down2 = DownReshape(2)
        self.down4 = DownReshape(4)
        self.down8 = DownReshape(8)

        Chans = [16, 32, 64, 128]
        self.sigmoid = nn.Sigmoid()

        self.BL1_1 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2] + Chans[3], Chans[0], 3, 1, True)
        self.BL2_1 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2] + Chans[3], Chans[1], 3, 1, True)
        self.BL3_1 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2] + Chans[3], Chans[2], 3, 1, True)
        self.BL4_1 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2] + Chans[3], Chans[3], 3, 1, True)

        self.BL1_2 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2] + Chans[3], Chans[0], 3, 1, True)
        self.BL2_2 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2] + Chans[3], Chans[1], 3, 1, True)
        self.BL3_2 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2] + Chans[3], Chans[2], 3, 1, True)

        self.BL1_3 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2], Chans[0], 3, 1, True)
        self.BL2_3 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2], Chans[1], 3, 1, True)
        self.BL3_3 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2], Chans[2], 3, 1, True)

        self.BL1_4 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2], Chans[0], 3, 1, True)
        self.BL2_4 = DecoderConvlution(Chans[0] + Chans[1] + Chans[2], Chans[1], 3, 1, True)

        self.BL1_5 = DecoderConvlution(Chans[0] + Chans[1], Chans[0], 3, 1, True)
        self.BL2_5 = DecoderConvlution(Chans[0] + Chans[1], Chans[1], 3, 1, True)

        self.BL1_6 = DecoderConvlution(Chans[0] + Chans[1], Chans[0], 3, 1, True)

        self.conv_out = Convlutionlayer(Chans[0], 1, 1, 1, True)
        self.high_freq_enhance = HighFreqBranch(Chans[0])

    def forward(self, x):
        # Stage1
        BL1_1 = self.BL1_1(
            torch.cat([x[0], self.up_eval2(x[0], x[1]), self.up_eval4(x[0], x[2]), self.up_eval8(x[0], x[3])], 1))
        BL2_1 = self.BL2_1(
            torch.cat([self.down2(x[1], x[0]), x[1], self.up_eval2(x[1], x[2]), self.up_eval4(x[1], x[3])], 1))
        BL3_1 = self.BL3_1(
            torch.cat([self.down4(x[2], x[0]), self.down2(x[2], x[1]), x[2], self.up_eval2(x[2], x[3])], 1))
        BL4_1 = self.BL4_1(
            torch.cat([self.down8(x[3], x[0]), self.down4(x[3], x[1]), self.down2(x[3], x[2]), x[3]], 1))
        # Stage2_1
        BL1_2 = self.BL1_2(
            torch.cat([BL1_1, self.up_eval2(BL1_1, BL2_1), self.up_eval4(BL1_1, BL3_1), self.up_eval8(BL1_1, BL4_1)], 1))
        BL2_2 = self.BL2_2(
            torch.cat([self.down2(BL2_1, BL1_1), BL2_1, self.up_eval2(BL2_1, BL3_1), self.up_eval4(BL2_1, BL4_1)], 1))
        BL3_2 = self.BL3_2(
            torch.cat([self.down4(BL3_1, BL1_1), self.down2(BL3_1, BL2_1), BL3_1, self.up_eval2(BL3_1, BL4_1)], 1))
        # Stage2_2
        BL1_3 = self.BL1_3(
            torch.cat([BL1_2, self.up_eval2(BL1_2, BL2_2), self.up_eval4(BL1_2, BL3_2)], 1))
        BL2_3 = self.BL2_3(
            torch.cat([self.down2(BL2_2, BL1_2), BL2_2, self.up_eval2(BL2_2, BL3_2)], 1))
        BL3_3 = self.BL3_3(
            torch.cat([self.down4(BL3_2, BL1_2), self.down2(BL3_2, BL2_2), BL3_2], 1))
        # Stage3_1
        BL1_4 = self.BL1_4(
            torch.cat([BL1_3, self.up_eval2(BL1_3, BL2_3), self.up_eval4(BL1_3, BL3_3)], 1))
        BL2_4 = self.BL2_4(
            torch.cat([self.down2(BL2_3, BL1_3), BL2_3, self.up_eval2(BL2_3, BL3_3)], 1))
        # Stage3_2
        BL1_5 = self.BL1_5(
            torch.cat([BL1_4, self.up_eval2(BL1_4, BL2_4)], 1))
        BL2_5 = self.BL2_5(
            torch.cat([self.down2(BL2_4, BL1_4), BL2_4], 1))
        # Stage4
        BL1_6 = self.BL1_6(
            torch.cat([BL1_5, self.up_eval2(BL1_5, BL2_5)], 1))

        # 加入细节增强
        enhanced = self.high_freq_enhance(BL1_6)

        output = self.sigmoid(self.conv_out(enhanced))
        return output


class MTEFuse_model(nn.Module):
    def __init__(self, in_c=1, out_c=64):
        super(MTEFuse_model, self).__init__()
        self.encoder = MTEFuse_Encoder(in_c=in_c, out_c=out_c)
        self.fuses1 = MTEFuse_Infuse(16 * 2, 16)
        self.fuses2 = MTEFuse_Infuse(32 * 2, 32)
        self.fuses3 = MTEFuse_Infuse(64 * 2, 64)
        self.fuses4 = MTEFuse_Infuse(128 * 2, 128)
        self.decoder = MTEFuse_Decoder()

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x11 = self.fuses1(torch.concat((x1[0], x2[0]), dim=1))
        x22 = self.fuses2(torch.concat((x1[1], x2[1]), dim=1))
        x33 = self.fuses3(torch.concat((x1[2], x2[2]), dim=1))
        x44 = self.fuses4(torch.concat((x1[3], x2[3]), dim=1))
        out = self.decoder([x11, x22, x33, x44])
        return out


if __name__ == '__main__':
    MTEFuse = MTEFuse_model(in_c=1, out_c=64)
    data1 = torch.rand((3, 1, 640, 480))
    data2 = torch.rand((3, 1, 640, 480))
    infuse_data = MTEFuse(data1, data2)
