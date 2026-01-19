import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
    def forward(self,x):
        return self.mlp(x)
 
class masked_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(masked_gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support, mask = None):
        out = [x]
        for a in support:
            if mask is not None:
                a = a * mask
            x1 = self.nconv(x,a) #对于每个support 进行一次普通的图卷积
            out.append(x1)
            for k in range(2, self.order + 1):#然后再上一步卷积的结果再进一步迭代多阶进行图卷积，并且每阶的信息都存储为结果之一
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
def floyd_warshall_optimized(adj_mx, num_nodes, max_distance=511):
    # Initialize the shortest paths matrix with adjacency values
    sh_mx = np.where(adj_mx > 0, 1, max_distance)
    np.fill_diagonal(sh_mx, 0)

    # Use numpy to optimize the Floyd-Warshall algorithm
    for k in range(num_nodes):
        # Update the shortest paths with the new intermediate node k
        np.minimum(sh_mx, np.add.outer(sh_mx[:, k], sh_mx[k, :]), out=sh_mx)
        
    return sh_mx

def retain_top_k_neighbors(dtw_mx, k=5):
    # Argsort each row and take the indices of the top k highest values (excluding self)
    top_k_neighbors = np.argsort(-dtw_mx, axis=1)[:, :k]
    
    # Create a new matrix to hold the top k neighbors with similarity values
    top_k_similarity_mx = np.zeros_like(dtw_mx)
    
    for i in range(dtw_mx.shape[0]):
        # Set the top k similarities
        top_k_similarity_mx[i, top_k_neighbors[i]] = dtw_mx[i, top_k_neighbors[i]]
    
    return top_k_similarity_mx
