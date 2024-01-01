import torch
import torch.nn as nn
import torch.utils.checkpoint
from timm.models.layers import PatchEmbed, Mlp, DropPath
from timm.models.vision_transformer import PatchEmbed, Block
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv_c = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_s = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax()
    def forward(self, x_c, x_s):
        B, N, C = x_c.shape

        qkv_c = self.qkv_c(x_c).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_c, k_c, v_c = qkv_c.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.scale
        attn_c = attn_c.softmax(dim=-1)
        attn_c = self.attn_drop(attn_c)

        qkv_s = self.qkv_s(x_s).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, k_s, v_s = qkv_s.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.scale
        attn_s = attn_s.softmax(dim=-1)
        attn_s = self.attn_drop(attn_s)

        x_cs = (attn_c @ v_s).transpose(1, 2).reshape(B, N, C)
        x_sc = (attn_s @ v_c).transpose(1, 2).reshape(B, N, C)
        
        x = x_cs + x_sc
        x = self.softmax(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.ReLU()
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(3, 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2, 1, 3, 1, 1),
            
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.softmax = nn.Softmax()
    def forward(self, x_c, x_s):
        x_c = self.norm1(x_c)
        x_s = self.norm1(x_s)

        x = torch.cat((x_c.unsqueeze(1), x_s.unsqueeze(1)), dim=1)
        x = self.softmax(x)

        x_1 = self.drop_path1(self.ls1(self.attn(x_c, x_s)))
        x_1 = self.norm2(x_1)
        x_1 = torch.cat((x, x_1.unsqueeze(1)), dim=1)
        x_1 = self.cnn1(x_1).squeeze(1)

        x_2 = self.drop_path2(self.ls2(x_1))
        x_2 = torch.cat((x, x_2.unsqueeze(1)), dim=1)
        x_2 = self.cnn2(x_2).squeeze(1)
        
        return x_2
