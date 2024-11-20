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
    def __init__(self, num_nodes, node_dims, project_hidden, bias=True):
        """
            num_nodes: 그래프 노드의 개수
            node_dims: 각 node feature diemnsion
            project_hidden: dimension
        """
        super(ProjectLayer, self).__init__()
        print("GraphTransformer - Layer Module Calling")
        # each node input feature을 가지고 project hidden dimension으로 변환 하기 위해 wight초기화
        self.project_weights = nn.ParameterList([
            nn.Parameter(torch.randn(node_dims[i], project_hidden)) for i in range(num_nodes)
        ])
        #reset bias and weights
        self.bias_term = nn.Parameter(torch.zeros(project_hidden)) if bias else None
        self._initialize_weights()

    def forward(self, h):
        #h --> input features tenser, weight (nodes num and feature dimension)
        projected_h = [torch.matmul(h[i], self.project_weights[i]) for i in range(len(self.project_weights))]
        if self.bias_term is not None:
            projected_h = [proj + self.bias_term for proj in projected_h]
        return torch.stack(projected_h, dim=1) #특징 텐서 뭐 이런거

    def _initialize_weights(self):
        for weight in self.project_weights:
            std_dev = 1. / math.sqrt(weight.size(1))
            weight.data.uniform_(-std_dev, std_dev)

class LongTermLayerAttention(nn.Module): #노드 장기적 관계 학습 계층
    def __init__(self, num_nodes, node_dim, project_hidden, tsf_dim, tsf_mlp_hidden, depth,
                    heads, head_dim=64, tsf_dropout=0., emb_dropout=0., pool='mean', bias=True):
        super(LongTermLayerAttention, self).__init__()
        print("GraphTransformer - LTLA Calling")
        self.projection_layer = ProjectLayer(num_nodes, node_dim, project_hidden, bias=bias)
        self.attention_module = GraphTransformer(num_nodes, project_hidden, tsf_dim, tsf_mlp_hidden, depth, heads,
                                                    head_dim=head_dim, tsf_dropout=tsf_dropout, emb_dropout=emb_dropout, pool=pool)

    def forward(self, h, edge_index):
        # h --> input feature tensor
        # edge_index --> graph edge info (엣지는 연관관계를 이야기함)
        projected_h = self.projection_layer(h)
        output_h = self.attention_module(projected_h, edge_index)
        return output_h

class GraphTransformer(nn.Module): #node relationship 학습 --> transformer module
    def __init__(self, num_nodes, node_dim, tsf_dim, tsf_mlp_hidden, depth, heads,
                    head_dim=64, tsf_dropout=0., emb_dropout=0., pool='mean'):
        super().__init__()
        print("GraphTransformer - GraphTransformer Module Calling")
        assert pool in {'mean', 'sum', 'max'} # checking  pooling function
        self.embedding_layer = nn.Linear(node_dim, tsf_dim) if node_dim != tsf_dim else nn.Identity()
        self.num_nodes = num_nodes
        self.position_embedding = nn.Parameter(torch.randn(1, num_nodes, tsf_dim))
        self.dropout_layer = nn.Dropout(emb_dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(tsf_dim, heads, head_dim, tsf_mlp_hidden, tsf_dropout)
            for _ in range(depth)
        ])
        self.pooling_mode = pool
        self.norm_layer = nn.LayerNorm(tsf_dim)

    def forward(self, x, edge_index):
        x = self.embedding_layer(x)
        x += self.position_embedding
        x = self.dropout_layer(x)
        for block in self.transformer_blocks:
            x = block(x, edge_index)
        if self.pooling_mode == 'mean':
            x = x.mean(dim=1)
        elif self.pooling_mode == 'sum':
            x = x.sum(dim=1)
        elif self.pooling_mode == 'max':
            x, _ = x.max(dim=1)
        return self.norm_layer(x)

class TransformerBlock(nn.Module): ## transformer에서 shows one layer
            # 1. multi-head attention
            # 2. feed-forward network
            # 3. skip connection
    def __init__(self, dim, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.attention_layer = PreNorm(dim, GraphAttention(dim, heads, head_dim, dropout))
        self.feedforward_layer = PreNorm(dim, FeedForward(dim, mlp_dim, dropout))

    def forward(self, x, edge_index):
        x = self.attention_layer(x, edge_index) + x
        x = self.feedforward_layer(x) + x
        return x

class GraphAttention(nn.Module):
    # learn the relationship between nodes
    # 1 query key value  입력된 텐서를 요 3개로 변환
    # 2 calc attention
    # 3. 그리고 가충지 적용해서 유사도 계산
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        self.inner_dim = head_dim * heads
        self.heads = heads
        self.scale_factor = head_dim ** -0.5
        self.query_key_layer = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.value_layer = nn.Linear(dim, self.inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, edge_index):
        batch_size, num_nodes, _ = x.shape
        query_key = self.query_key_layer(x).chunk(2, dim=-1)
        query, key = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), query_key)
        value = self.value_layer(x)
        value = rearrange(value, 'b n (h d) -> b h n d', h=self.heads)
        attention_scores = torch.einsum('bhid,bhjd->bhij', query, key) * self.scale_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.einsum('bhij,bhjd->bhid', attention_weights, value)
        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.output_layer(output)

class PreNorm(nn.Module):
    # 입력 데이터 정규화 -> 그리고 함수(fn) adapt
    def __init__(self, dim, fn):
        super().__init__()
        print("GraphTransformer - PreNorm Calling")
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class FeedForward(nn.Module):
    # transform feature 완전 연결 신경망  --> 과적합 방지 prevent over smoothing
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
        # self-attention -- >
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
    #transformer 를 block 여러개로 만든거 --> 그래서 깊이 있는 관계를 학습합
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, GraphAttention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, edge_index):
        for attn, ff in self.layers:
            x = attn(x, edge_index) + x
            x = ff(x) + x
        return x