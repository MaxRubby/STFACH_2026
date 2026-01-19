import numpy as np

def MSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse





def RMSE(y_true, y_pred, mask_val = np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        if not np.isnan(mask_val):
            mask &= y_true.ge(mask_val)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred, mask_val = np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        if not np.isnan(mask_val):
            mask &= y_true.ge(mask_val)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae
     


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def MAPE_filtered(y_true, y_pred, threshold=5.0, null_val=0):
    """
    MAPE with filtering: only calculate MAPE for samples where y_true >= threshold
    This reduces the impact of small values on MAPE calculation
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        
        # Additional filter: only include samples where y_true >= threshold
        mask = mask & (y_true >= threshold)
        
        if np.sum(mask) == 0:
            return 0.0
        
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def sMAPE(y_true, y_pred, null_val=0):
    """
    Symmetric MAPE: 2 * |y_pred - y_true| / (|y_true| + |y_pred|) * 100%
    More balanced than MAPE, less sensitive to small values
    Range: [0, 200%]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        
        # Avoid division by zero
        smape = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        smape = np.nan_to_num(mask * smape)
        return np.mean(smape) * 100


def WAPE(y_true, y_pred, null_val=0):
    """
    Weighted Absolute Percentage Error (WAPE)
    Also known as MAE/Mean ratio: sum(|y_pred - y_true|) / sum(|y_true|) * 100%
    Less sensitive to small values than MAPE
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        
        masked_error = np.abs(y_pred - y_true) * mask
        masked_true = np.abs(y_true) * mask
        
        sum_error = np.sum(masked_error)
        sum_true = np.sum(masked_true)
        
        if sum_true == 0:
            return 0.0
        
        wape = sum_error / sum_true
        return wape * 100


 

def RMSE_MAE_MAPE(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )


def ALL_METRICS(y_true, y_pred, filter_threshold=0.001):
    """
    Calculate all 10 metrics:
    1. RMSE
    2. MAE  
    3. MAPE
    4. sMAPE
    5. WAPE
    6. MAPE_filtered (threshold)
    7. MAPE_filtered (threshold * 10)
    8. MAPE_filtered (threshold * 100)
    9. MAPE_filtered (threshold * 1000)
    10. MAPE_filtered (threshold * 10000)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        filter_threshold: Base threshold for MAPE filtering (default 0.001)
    
    Returns:
        Tuple of 10 metrics: (rmse, mae, mape, smape, wape, 
                              mape_fil1x, mape_fil10x, mape_fil100x, mape_fil1000x, mape_fil10000x)
    """
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
        sMAPE(y_true, y_pred),
        WAPE(y_true, y_pred),
        MAPE_filtered(y_true, y_pred, threshold=filter_threshold),
        MAPE_filtered(y_true, y_pred, threshold=filter_threshold * 10),
        MAPE_filtered(y_true, y_pred, threshold=filter_threshold * 100),
        MAPE_filtered(y_true, y_pred, threshold=filter_threshold * 1000),
        MAPE_filtered(y_true, y_pred, threshold=filter_threshold * 10000),
    )

 

 
 