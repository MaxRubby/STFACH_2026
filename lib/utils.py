import numpy as np
import torch
import pickle
import random
import os
import json
import torch.nn as nn
from scipy.stats.distributions import norm
import torch.nn.functional as F
def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
 
 
def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()
 
def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
     

def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)

  