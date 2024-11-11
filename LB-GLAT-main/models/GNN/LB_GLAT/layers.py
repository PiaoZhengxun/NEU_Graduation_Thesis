# -*- coding: utf-8 -*-


import math
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from torch import LongTensor, Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ProjectLayer(nn.Module):
    def __init__(self, num_patches, patches_dim,  project_hidden, bias=True):
        super(ProjectLayer, self).__init__()
        # Projection
        # Hl*W  (batch, layer_hidden) * (layer_hidden, project_hidden)
        self.proj_ws = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(patches_dim[i], project_hidden)) for i in range(num_patches)  # (layer_hidden, project_hidden)
        ])
        self.reset_parameters()

    def forward(self, h):
        # Projection
        proj_h = []
        for i in range(len(self.proj_ws)):
            proj_h.append(torch.matmul(h[i], self.proj_ws[i]))  # (batch, project_hidden)
        h = torch.stack(proj_h, dim=1)  # (batch, layer_num, project_hidden)
        return h

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.proj_ws[0].size(1))
        for paras in self.proj_ws:
            paras.data.uniform_(-stdv1, stdv1)  # Randomization parameters


class LongTermLayerAttention(nn.Module):
    def __init__(self, num_patches, patches_dim,  project_hidden, tsf_dim, tsf_mlp_hidden, depth,
                 heads, head_dim=64, tsf_dropout=0., vit_emb_dropout=0., pool='cls', bias=True):
        super(LongTermLayerAttention, self).__init__()
        self.project = ProjectLayer(num_patches, patches_dim,  project_hidden, bias=bias)
        self.vit = VIT(num_patches, project_hidden, tsf_dim, tsf_mlp_hidden, depth, heads,
                       head_dim=head_dim, tsf_dropout=tsf_dropout, vit_emb_dropout=vit_emb_dropout, pool=pool)

    def forward(self, h):  # layer_num, (batch, layers_hidden)
        h = self.project(h)  # (batch, layer_num, project_hidden)
        h = self.vit(h)  # h(batch, tsf_dim)
        return h



class VIT(nn.Module):
    def __init__(self, num_patches, patch_dim, tsf_dim, tsf_mlp_hidden, depth, heads,
                 head_dim=64, tsf_dropout=0., vit_emb_dropout=0., pool='cls'):
        super().__init__()
        assert pool in {'cls', 'mean', 'last'}, \
            'VIT: Pool type must be either cls (cls token) or mean (mean pooling) or last (last token)'

        project_in = not (patch_dim == tsf_dim)  # If the patch_dim is not equal to tsf_dim, we need to convert the dimension
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, tsf_dim),
        ) if project_in else nn.Identity()

        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(
            torch.randn(1, (num_patches + 1) if pool == 'cls' else num_patches, tsf_dim))  # num_patches + 1增加的是cls
        self.cls_token = nn.Parameter(torch.randn(1, 1, tsf_dim))
        self.dropout = nn.Dropout(vit_emb_dropout)

        self.transformer = Transformer(tsf_dim, depth, heads, head_dim, tsf_mlp_hidden, tsf_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.norm = nn.Sequential(
            nn.LayerNorm(tsf_dim),
        )

    def forward(self, h):
        h = self.to_patch_embedding(h)  # h(batch, num_patches, tsf_dim)
        b, n, _ = h.shape  ## n is the number of sequences (equivalent to the number of layers of LayerAttention, in fact, you can also increase your own characteristics Layer + 1)
        assert n == self.num_patches, 'VIT: Layer number must be equal to the patch number of VIT'

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # Add the data of each batch to CLS
            h = torch.cat((cls_tokens, h), dim=1)
        h += self.pos_embedding  # h(batch, num_patches[+1], tsf_dim)
        h = self.dropout(h)

        h = self.transformer(h)  # h(batch, num_patches[+1], tsf_dim)

        if self.pool == 'mean':
            h = h.mean(dim=1)
        elif self.pool == 'last':
            h = h[:, -1]
        else:
            h = h[:, 0]

        h = self.to_latent(h)  # h(batch, tsf_dim)
        return self.norm(h)  # h(batch, tsf_dim)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


