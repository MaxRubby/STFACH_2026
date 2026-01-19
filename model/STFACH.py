import sys
# sys.path.append('..\\')
import numpy as np
import model.HeteroGCNs as HeteroGCNs
from lib.utils import StandardScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
from vector_fields import *
from lib.astancde_utils import *
import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 已注释：强制同步会严重降低GPU利用率

class STFACH(nn.Module):
    def __init__(self, num_nodes, input_dim=3, output_channels=1, in_steps = 12, out_steps = 12, embed_dim = 64, num_layers = 3, add_day_in_week = True,steps_per_day=288, add_time_in_day=True, hid_dim = 64, hid_hid_dim = 64, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), atol=1e-9, rtol=1e-7, solver='euler',  adj_mx = None, adj_semx = None, use_gated_fusion=False, output_activation='none', g_eps=5, s_eps=5):
        
        super(STFACH, self).__init__()
        self.adj_mx = adj_mx
        self.adj_semx = adj_semx
        self.atol = atol
        self.rtol = rtol
        self.solver =solver
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.input_channels = input_dim
        self.hidden_dim = hid_dim
        self.output_dim = output_channels
        self.horizon = out_steps
        self.lag = in_steps
        self.num_layers =  num_layers
        self.hid_hid_dim = hid_hid_dim
        self.hid_dim = hid_dim
        self.embed_dim = embed_dim
        self.add_day_in_week = add_day_in_week
        self.add_time_in_day = add_time_in_day
        #=============ACL======================
        self.feature_dim = 2  
        self.times = torch.linspace(0,self.lag -1, self.lag).to(device)
        self.func_attn = STAttnFunc(input_channels=self.embed_dim, hidden_channels=self.hid_dim,hidden_hidden_channels=self.hid_hid_dim, model_dim = self.embed_dim, num_hidden_layers = 3, feed_forward_dim=256, num_heads=4, drop=0.1)
        self.coeff_embedding = TokenEmbedding(self.feature_dim, self.embed_dim)
        self.ln = nn.LayerNorm(self.hidden_dim) #对cde的结果进行norm 
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.time_embed = 24  
        self.adp_emb_dim = 80
        self.model_dim = self.adp_emb_dim +  3 * self.time_embed  

        self.embed_layer = DataEmbedding(
            self.input_dim, self.output_dim, self.time_embed, self.adp_emb_dim, self.lag,  self.num_nodes,drop=0.,
            add_time_in_day= self.add_day_in_week, add_day_in_week=self.add_day_in_week, steps_per_day= steps_per_day, device=device,
        )

        self.output_proj = nn.Linear(self.lag * self.model_dim, self.horizon * self.output_dim)
        self.in_proj = nn.Linear(self.lag * self.model_dim, (self.lag - 1) * self.embed_dim)
        self.time_proj = nn.Linear(self.lag - 1, (self.lag - 1) * (self.lag - 1))
        self.time_proj2 = nn.Linear(self.lag - 1, (self.lag - 1) * (self.lag - 1))
        self.time_proj3 = nn.Linear(self.lag - 1, (self.lag - 1) * (self.lag - 1))
        self.time_proj4 = nn.Linear(self.lag - 1, (self.lag - 1) * (self.lag - 1))
        #---------------------------------HeteroGCNs-------------------------------------------------

        self.use_gated_fusion = use_gated_fusion
        
        if self.use_gated_fusion:
            self.gate_fc = nn.Sequential(
                nn.Linear(self.horizon * self.num_nodes * self.output_dim * 2, 
                         self.horizon * self.num_nodes * self.output_dim),
                nn.ReLU(),
                nn.Linear(self.horizon * self.num_nodes * self.output_dim, 
                         self.horizon * self.num_nodes * self.output_dim),
                nn.Sigmoid()
            )
        else:
            self.weight_acl = nn.Parameter(torch.Tensor([0.5]))   
            self.weight_hetero = nn.Parameter(torch.Tensor([0.5]))   
        
        # 动态计算 HeteroGCNs 的输入维度
        # embed_layer 输出维度: (B, T, N, model_dim)
        # model_dim = adp_emb_dim + 3 * time_embed = 80 + 3 * 24 = 152
        hetero_in_dim = self.model_dim  # 使用 model_dim 作为输入维度
        
        self.HeteroGCNs = HeteroGCNs.HeteroGCNs(device=device, in_dim=hetero_in_dim, num_nodes=self.num_nodes,
                                 out_dim=self.horizon * self.output_dim,
                                 out_steps=self.horizon,
                                 output_channels=self.output_dim,
                                 adj_mx=adj_mx, adj_semx=adj_semx,
                                 output_activation=output_activation,
                                 g_eps=g_eps, s_eps=s_eps)
    def forward(self, ori_x, coeffs, return_spline=False):
        B,T,N,D = ori_x.shape
        x_emb = self.embed_layer(ori_x)  
        x_in = self.in_proj(x_emb.permute(0,2,1,3).reshape(B,N,-1)).reshape(B, N, (T - 1), -1)
 
        _, x2,x3,x4 = coeffs#BNTD
        x2 = self.coeff_embedding(x2[:, :, :, :self.feature_dim])
        x3 = self.coeff_embedding(x3[:, :, :, :self.feature_dim])
        x4 = self.coeff_embedding(x4[:, :, :, :self.feature_dim])
        x1 = self.time_proj(x_in.transpose(-1,-2)).reshape(B,N,self.hid_dim, T-1, T-1).transpose(2,-1)
        x2 = self.time_proj2(x2.transpose(-1,-2)).reshape(B,N,self.hid_dim, T-1, T-1).transpose(2,-1)
        x3 = self.time_proj3(x3.transpose(-1,-2)).reshape(B,N,self.hid_dim, T-1, T-1).transpose(2,-1)
        x4 = self.time_proj4(x4.transpose(-1,-2)).reshape(B,N,self.hid_dim, T-1, T-1).transpose(2,-1)
        coeffs_transformed = (x1,x2,x3,x4)  
        spline = controldiffeq.NaturalCubicSpline(self.times, coeffs_transformed)
        
        # 可视化模式：返回原始系数和变换后的系数
        if return_spline:
            original_coeffs = coeffs  # 原始系数（未变换）
            # import pdb; pdb.set_trace()
            return spline, original_coeffs, coeffs_transformed
        h0 = x_in
        z0 = x_in
        option = {'interp':'linear'}
        with torch.enable_grad():
            z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                    h0=h0,
                                    z0=z0,
                                    func_f=self.func_attn,
                                    t=self.times,
                                    method=self.solver,
                                    atol=self.atol,
                                    rtol=self.rtol,
                                    options=option
                                    # ,adjoint_params = list(self.parameters())
                                    )
        z_t = z_t.permute(2,0,1,3)
        z_T = z_t[-1:,...].transpose(0,1)  
        z_T = self.ln(z_T)  
        out_acl = self.end_conv(z_T).reshape(z_T.shape[0], self.horizon, self.num_nodes, -1)
        out_hetero = self.HeteroGCNs(x_emb)
        if self.use_gated_fusion:
            batch_size = out_acl.shape[0]
            out_acl_flat = out_acl.reshape(batch_size, -1)
            out_hetero_flat = out_hetero.reshape(batch_size, -1)
            concat_features = torch.cat([out_acl_flat, out_hetero_flat], dim=-1)
            gate = self.gate_fc(concat_features)
            gate = gate.reshape(batch_size, self.horizon, self.num_nodes, self.output_dim)
            output = gate * out_acl + (1 - gate) * out_hetero
        else:
            output = self.weight_acl * out_acl + self.weight_hetero * out_hetero
        
        return output #Batch, out_step, nodes, out_dim
