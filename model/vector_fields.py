from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------------------ACL------------------------------
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, return_attn=False):
        batch_size = query.shape[0]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim) #head_dim=model_dim/head_dim

        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)
        
        if return_attn:
            # 将attn_score重组回原始batch维度
            attn_score = torch.cat(torch.split(attn_score, batch_size, dim=0), dim=-1)
            return out, attn_score
        return out

class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=256, num_heads=4, dropout=0.1, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ELU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout) #0.1
        self.dropout2 = nn.Dropout(dropout) #0.1


    def forward(self, x, dim=-2, return_attn=False):  
        x = x.transpose(dim, -2)  
        residual = x
        if return_attn:
            out, attn_weights = self.attn(x, x, x, return_attn=True)  # (batch_size, ..., length, model_dim)
        else:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        if return_attn:
            return out, attn_weights
        return out
  
class STAttnFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, model_dim, num_hidden_layers, control_dim = 4, feed_forward_dim=256, num_heads=4, drop=0.1, mask=False):
        super(STAttnFunc, self).__init__()

        self.model_dim = model_dim
        self.control_dim = control_dim
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.elu = torch.nn.ReLU(inplace=True)
        self.ln = torch.nn.LayerNorm(hidden_channels * input_channels)
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * input_channels) #32,32*4  -> # 32,32,4 
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, drop)
                for _ in range(num_hidden_layers)
            ]
        )

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, drop)
                for _ in range(num_hidden_layers)
            ]
        )
        self.conv = nn.Conv2d(in_channels=hidden_channels,  out_channels=hidden_channels * hidden_channels, kernel_size=(1, 1),bias=True)  
        
    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z, return_attn=False):
        temporal_attns = []
        spatial_attns = []
        
        for attn in self.attn_layers_t:
            if return_attn:
                z, t_attn = attn(z, dim=2, return_attn=True)
                temporal_attns.append(t_attn)
            else:
                z = attn(z, dim=2)
        
        for attn in self.attn_layers_s:
            if return_attn:
                z, s_attn = attn(z, dim=1, return_attn=True)
                spatial_attns.append(s_attn)
            else:
                z = attn(z, dim=1)
        
        z = self.conv(z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)  
        
        if return_attn:
            return z, temporal_attns, spatial_attns
        return z
  