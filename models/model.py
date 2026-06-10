import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers import *
from einops import rearrange


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]


def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W],device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res


def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


class Attention(nn.Module):
    def __init__(self, channel, win_size, num_heads, bias):
        super(Attention, self).__init__()

        self.channel = channel
        self.win_size = win_size
        self.num_heads = num_heads
        self.head_dim = channel // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.ln = LayerNorm2d(channel)
        self.qkv  = nn.Conv2d(channel, channel*3, kernel_size=1, stride=1, padding=0, bias=bias)

        self.fc = nn.Sequential(
            nn.Linear(win_size**2, win_size**2//64, bias=bias),
            nn.GELU(),
            nn.Linear(win_size**2//64, win_size**2, bias=bias)
        )
        self.sigmoid = nn.Sigmoid()

        self.ln_q = LayerNorm2d(channel)

        self.maxpool = nn.MaxPool2d(4, 4)
        self.proj = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = self.ln(x)

        x_win, win_nums = window_partitionx(x, self.win_size) # (win_num*b), c, win_h, win_w

        qkv = self.qkv(x_win)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        q_flatten = torch.flatten(q, -2)
        q_flatten = self.fc(q_flatten)
        q_ = q_flatten.reshape(-1, c, self.win_size, self.win_size)
        q_ = self.sigmoid(q_)
        q = torch.mul(q, q_)
        q = self.ln_q(q)
       
        k = self.maxpool(k)
        v = self.maxpool(v)
    
        q = rearrange(q, 'b (n c) h w -> b n (h w) c', n=self.num_heads) # (win_num*b), heads, (win_h*win_w), (c/num_heads)
        k = rearrange(k, 'b (n c) h w -> b n (h w) c', n=self.num_heads) # (win_num*b), heads, (win_h*1), (c/num_heads)
        v = rearrange(v, 'b (n c) h w -> b n (h w) c', n=self.num_heads) # (win_num*b), heads, (win_h*1), (c/num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b n (h w) c -> b (n c) h w', w=self.win_size) # (win_num*b), c, win_h, win_w

        out = self.proj(out) # (win_num*b), c, win_h, win_w
        out = window_reversex(out, self.win_size, h, w, win_nums) # b, c, h, w

        return out
        
    def flops(self, H, W):

        flops = 0
        B = (H / self.win_size) * (W / self.win_size)
        # pool = self.win_size / self.pool_size
        # ln
        flops += self.channel * H * W
        # qkv
        flops += (self.channel * self.channel * 3) * (self.win_size * self.win_size) * B
        # fc
        flops += (self.win_size**2 * self.win_size**2//64) * self.channel * B * 2
        # ln_q
        flops += self.channel * H * W
        # q @ k^T
        flops += (self.win_size * self.win_size) * self.channel * (self.win_size//4 * self.win_size//4) * B
        # # attn @ v
        flops += (self.win_size * self.win_size) * (self.win_size//4 * self.win_size//4) * self.channel * B
        #proj
        flops += (self.channel * self.channel) * (self.win_size * self.win_size) * B

        return flops


class FFN(nn.Module):
    def __init__(self, channel, bias):
        super().__init__()
        self.channel = channel

        self.ln = LayerNorm2d(channel)
        self.conv = nn.Conv2d(channel, channel*8, 1, 1, 0, bias=bias)
        self.gelu = nn.GELU()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(channel*4, channel, 1, 1, 0, bias=bias)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channel*4, channel*4, 3, 1, 1, groups=channel*4, bias=bias),
            nn.GELU(),
            nn.Conv2d(channel*4, channel, 1, 1, 0, bias=bias)
        )

    def forward(self, x):

        x = self.conv(self.ln(x))
        x = self.gelu(x)

        x1, x2 = torch.chunk(x, chunks=2, dim=1) # b, c*4, h, w

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        out = x1 + x2

        return out
    
    def flops(self, H, W):

        flops = 0
        # norm
        flops += self.channel * H * W
        # conv c->8c
        flops += (self.channel * self.channel * 8) * H * W
        # gelu
        flops += self.channel * 8 * H * W
        # branch1.conv
        flops += (self.channel * 4 * self.channel) * H * W
        # branch2.dpw
        flops += (self.channel * 4 * 3 * 3) * H * W
        # branch2.gelu
        flops += self.channel * 4 * H * W
        # branch.conv c 4c->c
        flops += (self.channel * 4 * self.channel) * H * W


        return flops

##########################################################################
# block
class Block(nn.Module):
    def __init__(self, channel, win_size, num_heads, bias=False):
        super(Block, self).__init__()

        self.attention = Attention(channel=channel, win_size=win_size, num_heads=num_heads, bias=bias)
        self.ffn = FFN(channel=channel, bias=bias)

    def forward(self, x):
        x = self.attention(x) + x

        x = self.ffn(x) + x
        return x 

    def flops(self, H, W):
        
        flops = 0
        # attention
        flops += self.attention.flops(H, W)
        # ffn
        flops += self.ffn.flops(H, W)

        return flops


##########################################################################
# layer
class Layer(nn.Module):
    def __init__(self, channel, win_size, num_heads, depth, bias=False):
        super(Layer, self).__init__()
        self.layers = nn.ModuleList([
            Block(channel=channel, win_size=win_size, num_heads=num_heads, bias=bias) for _ in range(depth)
        ])

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x

    def flops(self, H, W):

        flops = 0
        for block in self.layers:
            flops += block.flops(H, W)

        return flops


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, channel, bias):
        super(OverlapPatchEmbed, self).__init__()
        self.in_c = in_c
        self.channel = channel
        self.proj = nn.Conv2d(in_c, channel, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act_layer = nn.LeakyReLU(True)

    def forward(self, x):
        x = self.proj(x)
        x = self.act_layer(x)

        return x

    def flops(self, H, W):

        flops = 0
        # proj
        flops += (self.in_c * self.channel * 3 * 3 + self.channel) * H * W

        return flops


##########################################################################
## Resizing modules
class DownSample(nn.Module):
    def __init__(self, channel, bias):
        super(DownSample, self).__init__()
        self.channel = channel
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel*2, kernel_size=3, stride=2, padding=1, bias=bias), 
            nn.ReLU(True)
        )

    def forward(self, x):
        
        return self.body(x)

    def flops(self, H, W):

        flops = 0
        # conv
        flops += (self.channel * self.channel * 2 * 3 * 3 + self.channel * 2) * H / 2 * W / 2 
        # gelu
        flops += self.channel * 2 * H / 2 * W / 2 

        return flops

class UpSample(nn.Module):
    def __init__(self, channel, bias):
        super(UpSample, self).__init__()
        self.channel = channel
        self.body = nn.Sequential(
            nn.Conv2d(channel*2, channel*4, kernel_size=1, bias=bias),
            nn.PixelShuffle(2), 
        )

    def forward(self, x):

        return self.body(x)

    def flops(self, H, W):

        flops = 0
        # conv
        flops += self.channel * 2 * self.channel * 4 * H * W

        return flops


class Net(nn.Module):
    def __init__(self, 
                inp_channels = 3, 
                out_channels = 3, 
                channel = 32, 
                win_size = [64,64,64], 
                heads = [1,2,4], 
                depth = [4,4,16],
                bias = False
        ):

        super(Net, self).__init__()

        self.channel = channel

        self.patch_embed = OverlapPatchEmbed(inp_channels, channel, bias=bias)

        ##### Detail Stage
        self.Layers = nn.ModuleList([
            Layer(channel=channel,   win_size=win_size[0], num_heads=heads[0], depth=depth[0], bias=bias),
            Layer(channel=channel*2, win_size=win_size[1], num_heads=heads[1], depth=depth[1], bias=bias),
            Layer(channel=channel*4, win_size=win_size[2], num_heads=heads[2], depth=depth[2], bias=bias),
            Layer(channel=channel*4, win_size=win_size[2], num_heads=heads[2], depth=depth[2], bias=bias),
            Layer(channel=channel*2, win_size=win_size[1], num_heads=heads[1], depth=depth[1], bias=bias),
            Layer(channel=channel,   win_size=win_size[0], num_heads=heads[0], depth=depth[0], bias=bias),
        ])

        self.Down = nn.ModuleList([
            DownSample(channel=channel,   bias=bias), 
            DownSample(channel=channel*2, bias=bias), 
        ])

        self.Up = nn.ModuleList([
            UpSample(channel=channel,   bias=bias), 
            UpSample(channel=channel*2, bias=bias), 
        ])

        self.output = nn.Conv2d(channel, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):

        z = self.patch_embed(inp_img)

        ### Encoder 1
        res1 = self.Layers[0](z)
        z = self.Down[0](res1)

        ### Encoder 2
        res2 = self.Layers[1](z)
        z = self.Down[1](res2)

        ### bottleneck
        z = self.Layers[2](z)
        z = self.Layers[3](z)

        ### Decoder 2
        z = self.Up[1](z)
        z = self.Layers[4](z + res2)

        ### Decoder 1
        z = self.Up[0](z)
        z = self.Layers[5](z + res1)
    
        out = self.output(z)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        return out + inp_img

    def flops(self, x):
        _, _, H, W = x.shape

        flops = 0
        # patch_embed
        flops += self.patch_embed.flops(H, W)
        # layer[0]
        flops += self.Layers[0].flops(H, W)
        # print('layer.0', self.Layers[0].flops(H, W) / 1e9)
        # layer[1]
        flops += self.Layers[1].flops(H / 2, W/ 2)
        # print('layer.1', self.Layers[1].flops(H / 2, W/ 2) / 1e9)
        # layer[2]
        flops += self.Layers[2].flops(H / 4, W/ 4)
        # print('layer.2', self.Layers[2].flops(H / 4, W/ 4) / 1e9)
        # layer[3]
        flops += self.Layers[3].flops(H / 4, W / 4)
        # print('layer.3', self.Layers[3].flops(H / 4, W / 4) / 1e9)
        # layer[4]
        flops += self.Layers[4].flops(H / 2, W / 2)
        # print('layer.4', self.Layers[4].flops(H / 2, W / 2) / 1e9)
        # layer[5]
        flops += self.Layers[5].flops(H, W)
        # print('layer.5', self.Layers[5].flops(H, W) / 1e9)
        # down
        flops += self.Down[0].flops(H, W)
        flops += self.Down[1].flops(H / 2, W / 2)
        # up
        flops += self.Up[1].flops(H / 4, W / 4)
        flops += self.Up[0].flops(H / 2, W / 2)
        # out
        flops += (self.channel * 3 * 3 * 3 + 3) * H * W

        return flops
