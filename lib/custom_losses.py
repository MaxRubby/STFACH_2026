"""
Custom Loss Functions for Sparse Grid Data
Designed to be sensitive to both small and large values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss: log(cosh(pred - true))
    
    优点:
    - 对小误差和大误差都敏感
    - 比MSE更平滑，比MAE更可微
    - 对异常值不如MSE敏感，但比Huber更关注小误差
    
    适用场景: 稀疏数据，需要平衡小值和大值的预测
    
    数值稳定性:
    - 使用 log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2) 的稳定形式
    - 当 |x| > 10 时近似为 |x| 避免溢出
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()
    
    def forward(self, pred, true):
        error = pred - true
        # 数值稳定的 log(cosh(x)) 实现
        # log(cosh(x)) ≈ |x| + log(1 + exp(-2|x|)) - log(2)
        # 当 |x| 很大时，log(cosh(x)) ≈ |x| - log(2)
        abs_error = torch.abs(error)
        
        # 对于大误差，使用线性近似避免溢出
        # 对于小误差，使用精确公式
        loss = torch.where(
            abs_error < 10.0,
            torch.log(torch.cosh(torch.clamp(error, -10.0, 10.0))),
            abs_error - np.log(2.0)
        )
        
        return torch.mean(loss)
    
    def _get_name(self):
        return 'LogCoshLoss'


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss)
    
    优点:
    - 可以通过quantile参数控制对高估/低估的惩罚
    - quantile=0.5时等价于MAE
    - 适合不对称分布的数据
    
    参数:
    - quantile: 0.5表示中位数，<0.5惩罚低估，>0.5惩罚高估
    """
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, pred, true):
        errors = true - pred
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)
    
    def _get_name(self):
        return f'QuantileLoss(q={self.quantile})'


class MAPELoss(nn.Module):
    """
    MAPE Loss with epsilon to avoid division by zero
    
    优点:
    - 直接优化MAPE指标
    - 对小值和大值都按百分比惩罚
    
    缺点:
    - 对接近0的值可能不稳定
    
    参数:
    - epsilon: 避免除零的小常数
    """
    def __init__(self, epsilon=1.0):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, true):
        # 添加数值稳定性：避免除以接近0的值
        denominator = torch.abs(true) + self.epsilon
        loss = torch.abs(true - pred) / torch.clamp(denominator, min=self.epsilon)
        # 裁剪极端值避免NaN
        loss = torch.clamp(loss, max=10.0)  # 限制最大为1000%
        return torch.mean(loss) * 100
    
    def _get_name(self):
        return 'MAPELoss'


class AdaptiveHuberLoss(nn.Module):
    """
    Adaptive Huber Loss with value-dependent delta
    
    优点:
    - delta根据真实值自适应调整
    - 对小值使用小delta（更敏感），对大值使用大delta
    - 平衡了对不同值范围的关注
    
    参数:
    - base_delta: 基础delta值
    - scale_factor: delta随值增长的比例
    """
    def __init__(self, base_delta=1.0, scale_factor=0.1):
        super(AdaptiveHuberLoss, self).__init__()
        self.base_delta = base_delta
        self.scale_factor = scale_factor
    
    def forward(self, pred, true):
        # 根据真实值自适应调整delta
        delta = self.base_delta + self.scale_factor * torch.abs(true)
        # 限制delta的范围避免数值问题
        delta = torch.clamp(delta, min=0.1, max=100.0)
        
        error = pred - true
        abs_error = torch.abs(error)
        
        # Huber loss with adaptive delta
        quadratic = torch.min(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        
        # 检查NaN
        if torch.isnan(loss).any():
            print("Warning: NaN detected in AdaptiveHuberLoss, using fallback")
            loss = torch.abs(error)  # fallback to MAE
        
        return torch.mean(loss)
    
    def _get_name(self):
        return 'AdaptiveHuberLoss'


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss with value-dependent weights
    
    优点:
    - 对小值给予更高权重，避免被大值主导
    - 可以通过weight_type调整权重策略
    
    参数:
    - weight_type: 'inverse', 'log', 'sqrt'
    - epsilon: 避免除零
    """
    def __init__(self, weight_type='inverse', epsilon=1.0):
        super(WeightedMSELoss, self).__init__()
        self.weight_type = weight_type
        self.epsilon = epsilon
    
    def forward(self, pred, true):
        error = (pred - true) ** 2
        
        if self.weight_type == 'inverse':
            # 小值权重大，大值权重小
            weights = 1.0 / (torch.abs(true) + self.epsilon)
        elif self.weight_type == 'log':
            # 对数权重
            weights = 1.0 / (torch.log1p(torch.abs(true)) + self.epsilon)
        elif self.weight_type == 'sqrt':
            # 平方根权重
            weights = 1.0 / (torch.sqrt(torch.abs(true) + self.epsilon))
        else:
            weights = torch.ones_like(true)
        
        # 归一化权重
        weights = weights / torch.mean(weights)
        
        loss = error * weights
        return torch.mean(loss)
    
    def _get_name(self):
        return f'WeightedMSELoss({self.weight_type})'


class BalancedL1Loss(nn.Module):
    """
    Balanced L1 Loss
    
    优点:
    - 对小值和大值使用不同的权重
    - 通过alpha参数平衡小值和大值的重要性
    
    参数:
    - threshold: 区分小值和大值的阈值
    - alpha: 小值的权重倍数（>1表示更关注小值）
    """
    def __init__(self, threshold=10.0, alpha=2.0):
        super(BalancedL1Loss, self).__init__()
        self.threshold = threshold
        self.alpha = alpha
    
    def forward(self, pred, true):
        error = torch.abs(pred - true)
        
        # 小值区域
        small_mask = (torch.abs(true) < self.threshold).float()
        # 大值区域
        large_mask = 1.0 - small_mask
        
        # 对小值给予更高权重
        loss = self.alpha * error * small_mask + error * large_mask
        
        return torch.mean(loss)
    
    def _get_name(self):
        return 'BalancedL1Loss'


class CombinedLoss(nn.Module):
    """
    Combined Loss: MAE + MAPE + MSE
    
    优点:
    - 结合多个损失函数的优点
    - MAE关注绝对误差，MAPE关注相对误差，MSE惩罚大误差
    
    参数:
    - mae_weight: MAE的权重
    - mape_weight: MAPE的权重
    - mse_weight: MSE的权重
    - epsilon: MAPE中避免除零
    """
    def __init__(self, mae_weight=1.0, mape_weight=0.5, mse_weight=0.5, epsilon=1.0):
        super(CombinedLoss, self).__init__()
        self.mae_weight = mae_weight
        self.mape_weight = mape_weight
        self.mse_weight = mse_weight
        self.epsilon = epsilon
    
    def forward(self, pred, true):
        # MAE
        mae = torch.mean(torch.abs(pred - true))
        
        # MAPE
        mape = torch.mean(torch.abs((true - pred) / (torch.abs(true) + self.epsilon)))
        
        # MSE
        mse = torch.mean((pred - true) ** 2)
        
        # 组合损失
        loss = (self.mae_weight * mae + 
                self.mape_weight * mape + 
                self.mse_weight * mse)
        
        return loss
    
    def _get_name(self):
        return 'CombinedLoss'


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - inspired by Focal Loss for object detection
    
    优点:
    - 自动关注难预测的样本
    - gamma参数控制对困难样本的关注度
    
    参数:
    - gamma: 聚焦参数，越大越关注困难样本
    """
    def __init__(self, gamma=2.0):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, pred, true):
        mse = (pred - true) ** 2
        
        # 计算相对误差作为难度指标
        relative_error = torch.abs(pred - true) / (torch.abs(true) + 1.0)
        
        # Focal weight: 误差越大，权重越大
        focal_weight = (1 + relative_error) ** self.gamma
        
        loss = mse * focal_weight
        return torch.mean(loss)
    
    def _get_name(self):
        return 'FocalMSELoss'


class RMSLELoss(nn.Module):
    """
    Root Mean Squared Logarithmic Error (RMSLE)
    
    优点:
    - 对小值和大值都敏感（对数尺度）
    - 只惩罚低估，不惩罚高估（如果使用log(pred+1) - log(true+1)）
    - 适合值域跨度大的数据
    
    参数:
    - epsilon: 避免log(0)
    
    数值稳定性:
    - 使用 log1p(x) = log(1+x) 更稳定
    - 对负预测值进行截断处理
    """
    def __init__(self, epsilon=1.0):
        super(RMSLELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, true):
        # 确保预测值非负（对于流量数据）
        pred_clipped = torch.clamp(pred, min=0.0)
        true_clipped = torch.clamp(true, min=0.0)
        
        # 使用 log1p 更稳定
        log_pred = torch.log1p(pred_clipped + self.epsilon)
        log_true = torch.log1p(true_clipped + self.epsilon)
        
        loss = torch.sqrt(torch.mean((log_pred - log_true) ** 2) + 1e-8)
        return loss
    
    def _get_name(self):
        return 'RMSLELoss'


# 便捷函数：根据名称获取损失函数
def get_loss_function(loss_name, **kwargs):
    """
    根据名称获取损失函数
    
    Args:
        loss_name: 损失函数名称
        **kwargs: 损失函数的参数
    
    Returns:
        损失函数实例
    """
    loss_dict = {
        'logcosh': LogCoshLoss,
        'quantile': QuantileLoss,
        'mape': MAPELoss,
        'adaptive_huber': AdaptiveHuberLoss,
        'weighted_mse': WeightedMSELoss,
        'balanced_l1': BalancedL1Loss,
        'combined': CombinedLoss,
        'focal_mse': FocalMSELoss,
        'rmsle': RMSLELoss,
    }
    
    if loss_name.lower() not in loss_dict:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_dict.keys())}")
    
    return loss_dict[loss_name.lower()](**kwargs)
