import torch.nn as nn
import torch
class HisPattern(nn.Module):
    def __init__(self, dim = 4):
        super(HisPattern, self).__init__()
        self.weight_shift = nn.Parameter(torch.randn(dim))
    def forward(self, shifts):
        shifts = shifts.mean(axis = -2)
        weight_shift_normalized = torch.sigmoid(self.weight_shift)
        shifts_weighted = shifts * weight_shift_normalized.expand_as(shifts)
        shifts_weighted = shifts_weighted.sum(dim=-1).unsqueeze(-1)
        return shifts_weighted
