import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.hetero_utils import *

class HeteroGCNs(nn.Module):
    def __init__(self,  device, num_nodes, dropout=0.3, in_dim=64,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, g_eps = 5, s_eps = 5, adj_mx = None,adj_semx = None,supports = None, out_steps=12, output_channels=1, output_activation='none', is_metro = False):
        """
        Args:
            output_activation: 'sigmoid', 'relu', or 'none'
                - 'sigmoid': 输出范围[0,1]，适用于归一化后的数据
                - 'relu': 输出非负值，适用于流量等非负数据
                - 'none': 无激活函数，适用于大范围数值预测
        """
        super(HeteroGCNs, self).__init__()
        self.out_steps = out_steps
        self.output_channels = output_channels
        self.output_activation = output_activation
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.geo_gconv = nn.ModuleList()
        self.sem_gconv = nn.ModuleList()
        self.is_metro = is_metro

        adj_mx =torch.tensor( floyd_warshall_optimized(adj_mx, num_nodes = num_nodes)).float()
        self.geo_mask = (adj_mx <= g_eps).float().to(device)
        self.sem_mask = torch.tensor(retain_top_k_neighbors(adj_semx, k = s_eps)).bool().float().to(device)
        self.sem_mask = (~self.sem_mask.bool()).float()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports =  []
        receptive_field = 1
        self.supports_len = 0

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)  
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec3 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)  
        self.nodevec4 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1

        for b in range(blocks):
            additional_scope = kernel_size - 1  
            new_dilation = 1  
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,  
                                                   out_channels=dilation_channels, #32
                                                   kernel_size=(1,kernel_size),dilation=new_dilation)) #kernel_size 2， new_dilation 1  
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2  
                receptive_field += additional_scope  
                additional_scope *= 2  
            
                self.geo_gconv.append(masked_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                self.sem_gconv.append(masked_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
    def forward(self, input): # ! (B, C, N, T)
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0)) #避免空洞？所以需要填充  torch.Size([64, 2, 307, 13])
        else:
            x = input
        
        x = self.start_conv(x)  
        skip = 0
        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1) #relu 转负为正，softmax在列上进行softmax，统计每行的比例，也即每个节点与其他节点的关联性矩阵
            adp2 = F.softmax(F.relu(torch.mm(self.nodevec3, self.nodevec4)), dim=1) #relu 转负为正，softmax在列上进行softmax，统计每行的比例，也即每个节点与其他节点的关联性矩阵
            new_supports = self.supports + [adp]
            new_supports2 = self.supports + [adp2]

        # Temporal Aggregation layers
        for i in range(self.blocks * self.layers):
            residual = x  
            filter = self.filter_convs[i](residual)  
            filter = torch.tanh(filter)  
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate  
            s = x 
            s = self.skip_convs[i](s)  

            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.supports is not None:
                x_geo = self.geo_gconv[i](x, new_supports, self.geo_mask)  
                x_sem = self.sem_gconv[i](x, new_supports2, self.sem_mask)  
                x = x_geo+x_sem
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  
        x = F.relu(self.end_conv_1(x))  
        x = self.end_conv_2(x)  #batch, out_dim * out_steps, N, 1
        
        # 根据配置选择输出激活函数
        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'relu':
            x = F.relu(x) 
        # x.shape shmetro:torch.Size([16, 8, 288, 1]) 
        # Reshape output to (B, out_steps, N, output_channels)
        # x shape: (B, out_dim, N, 1) where out_dim = out_steps * output_channels
        # 此时
        if self.output_channels > 1 and self.out_steps > 1:
            B, _, N, _ = x.shape
            x = x.permute(0, 3, 2, 1)  # (B, 1, N, out_dim * out_steps)
            # Take the last timestep
            x = x[:, -1, :, :]  # (B, N, out_dim * out_steps)
            # Reshape to (B, out_steps, N, output_channels)
            x = x.reshape(B, N, self.out_steps, self.output_channels)
            x = x.permute(0, 2, 1, 3)  # (B, out_steps, N, output_channels)
        elif self.output_channels > 1: # output_channels > 1 & out_steps = 1 需要out_channels置换到最后一维
            x = x.permute(0, 3, 2, 1)
        # out_steps > 1 & output_channels = 1 的情况下直接返回，此时第二维度就是输出步长
        return x
