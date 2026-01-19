# sys.path.append('..')
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(
        self, input_dim, output_dim, embed_dim, adp_dim,lag, num_node, drop=0.,
        add_time_in_day=False, add_day_in_week=False, steps_per_day = 288, device=torch.device('cpu'),
    ):
        super().__init__()
        self.steps_per_day = steps_per_day
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adp_dim = adp_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.week_proj = nn.Linear(1, embed_dim)

        if self.add_time_in_day:
            self.tod_embedding = nn.Embedding(self.steps_per_day, embed_dim)
        if self.add_day_in_week:
            self.dow_embedding = nn.Embedding(7, embed_dim)
 
        if self.adp_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(lag, num_node, adp_dim))
        )

 
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.add_time_in_day > 0:
            tod = x[..., self.output_dim]
        if self.add_day_in_week > 0:
            dow = x[..., self.output_dim + 1]
        
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        
        if self.add_time_in_day > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.add_day_in_week > 0:
            dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)

        if self.adp_dim > 0: # self.adaptive_embedding:torch.Size([12, 307, 80])
            adp_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape)) #eg.(1,*self.adaptive_embedding.shape) => ((1, 12, 307, 80)) 所以这个embedding是跨batch共享的嵌入参数？bingo
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        return x 