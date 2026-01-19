import argparse
import copy
import random
import numpy as np
import scipy.sparse as sp

import pandas as pd
import os
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
SUPPORTS_WEIGHTS_ONLY = TORCH_VERSION >= (2, 0)

def safe_torch_load(path):
    """兼容不同 PyTorch 版本的 load 函数"""
    if SUPPORTS_WEIGHTS_ONLY:
        return torch.load(path, weights_only=False)
    else:
        return torch.load(path)
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
sys.path.append("..")
from lib.utils import (
    init_seed,
    print_log,
    set_cpu_num,
    CustomJSONEncoder, MaskedMAELoss,
)
from lib.metrics import RMSE_MAE_MAPE, ALL_METRICS
from lib.data_prepare import generate_data
from lib.custom_losses import get_loss_function
from model.STFACH import STFACH
from model.HisPatternLearner import HisPattern

def make_dir(name):
    if (os.path.exists(name)):
        print('has  save path')
    else:
        os.makedirs(name)

@torch.no_grad()
def eval_model(model, valset_loader, criterion, epoch, scheduler ,  shift_weight = None):
    model.eval()
    batch_loss_list = []
    valid_per_epoch = len(valset_loader)
    with torch.no_grad():
        for batch_idx,batch in enumerate(valset_loader):
            batch = tuple(b.to(DEVICE) for b in batch)
            x_batch,y_batch,  *coeff_batch= batch #BTND(3)
            shifts = x_batch[...,-1:]
            out_batch =  model(ori_x=x_batch, coeffs = coeff_batch)
            out_batch = SCALER.inverse_transform(out_batch)
            shifts = SCALER.inverse_transform(shifts)
            loss= criterion(out_batch, y_batch)
            batch_loss_list.append(loss.detach())
            
             
    epoch_loss = torch.stack(batch_loss_list).mean().item()
    scheduler.step(metrics=epoch_loss)
    
    return epoch_loss


@torch.no_grad()
def predict(model, testset_loader,   epoch = 0 ):
    model.eval()
    y = []
    out = []
    for batch_idx,batch in enumerate(testset_loader):

        batch = tuple(b.to(DEVICE) for b in batch)
        x_batch,y_batch,  *coeff_batch= batch #BTND(3)
        shifts = x_batch[..., -1:]

        out_batch =  model(ori_x=x_batch,   coeffs = coeff_batch)
        
        out_batch = SCALER.inverse_transform(out_batch)
        shifts = SCALER.inverse_transform(shifts)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)
    out = np.vstack(out) # (samples, out_steps, num_nodes)
    y = np.vstack(y)
    if out.shape[1] > 1:
        out = out.squeeze()
        y = y.squeeze()
    return y, out

  
def train_one_epoch(
    model,  trainset_loader, optimizer, scheduler, criterion,  clip_grad, epoch, mixup_num, log=None,w=None, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),   shift_weight = None
):
    global cfg , iter_count , target_length       
 

    model.train()

     
    batch_loss_list = []
    x_batch_accumulate = torch.tensor([], device=device)
    y_batch_accumulate = torch.tensor([], device=device)
    shifts_accumulate = torch.tensor([], device=device)
    

    for batch_idx,batch in enumerate(trainset_loader):
        
        lambd = np.random.beta(mix_alpha, mix_alpha) 

        if cfg["use_cl"]: 
            if (
                iter_count % cfg["cl_step_size"] == 0
                and target_length < cfg["out_steps"]
            ):
                target_length += 1
                print_log(f"CL target length = {target_length}", log=log)
            iter_count += 1

        batch = tuple(b.to(DEVICE) for b in batch)
        x_batch,y_batch,  *coeff_batch= batch #BTND(3)
        
        shifts =x_batch[..., -1:]

        if batch_idx > 1 and batch_idx % mixup_num == 0 :
            id_1 = np.arange(x_batch.shape[0])  
            x_1 = x_batch[id_1]
            y_1 = y_batch[id_1]
            coeffs_1 = tuple(t[id_1] for t in coeff_batch)  
            id_2 = torch.randperm(x_batch.shape[0])
            x_2 = x_batch[id_2]
            y_2 = y_batch[id_2]
            coeffs_2 = tuple(t[id_2] for t in coeff_batch)  

            mixup_Y = y_1 * lambd + y_2 * (1 - lambd)
            mixup_X = x_1 * lambd + x_2 * (1 - lambd)
            mixup_coeffs = tuple(t1 * lambd + t2 * (1 - lambd) for t1, t2 in zip(coeffs_1, coeffs_2))
            
            out_batch =  model(ori_x=mixup_X, coeffs = mixup_coeffs)  
             
        else:
            out_batch =  model(ori_x=x_batch, coeffs = coeff_batch)

        if  batch_idx % 50 == 0 and args.speed:
            model_time_s = time.time()       

        if  batch_idx % 50 == 0 and args.speed:
            model_time_e = time.time()     

        out_batch = SCALER.inverse_transform(out_batch)
        shifts = SCALER.inverse_transform(shifts)

        x_batch_accumulate = torch.cat((x_batch_accumulate, x_batch[:,:,:,:1].mean(axis = -2)), dim=0)
        y_batch_accumulate = torch.cat((y_batch_accumulate, y_batch[:,:,:,:1].mean(axis = -2)), dim=0)
        shifts_accumulate = torch.cat((shifts_accumulate, shifts[:,:,:,:1].mean(axis = -2)), dim=0)

         
        if batch_idx > 1 and batch_idx % mixup_num == 0 :
            loss = criterion(out_batch[:, : target_length, ...], mixup_Y[:, : target_length, ...])
        else:
            loss = criterion(out_batch[:, : target_length, ...], y_batch[:, : target_length, ...])
            
        
        batch_loss_list.append(loss.detach())
        train_per_epoch = len(trainset_loader)
        optimizer.zero_grad()
        if  batch_idx % 50 == 0 and args.speed:
            lb_s = time.time()   
  
        loss.backward()

        if  batch_idx % 50 == 0 and args.speed:
            lb_e = time.time()     
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        if  batch_idx % 50 == 0 and args.speed:
            opt_time_s = time.time()  
        optimizer.step()
        if  batch_idx % 50 == 0 and args.speed:
            opt_time_e = time.time() 
        if  batch_idx % 50 == 0:
            if args.speed:
                board_time_s =   time.time()
            if args.speed:           
                board_time_e = time.time()
            
            print_log(f"Train Epoch {epoch}: {batch_idx}/{len(trainset_loader)}  Loss: {loss.item()}  LR:{optimizer.param_groups[0]['lr']} ",log=log)
      
                    

    epoch_loss = torch.stack(batch_loss_list).mean().item()
    return epoch_loss 


 
def save_checkpoint( model,optimizer, epoch, min_var_loss, log = None):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'min_val_loss': min_var_loss
    }
    
    cache_path = log_dir + '/model_ckpt_{}.pth'.format(epoch)
    torch.save(state, cache_path)
    print_log(f"Saving current best model to {cache_path}", log=log)

def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    adj_mx,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
    cache_path = None,
    w=None,
    his_opt = None,
    his_learner = None,
    mixup_num=5,
    mix_alpha=2.0
):
    if cache_path is not None and args.cont > 0:
        model.load_state_dict(safe_torch_load(cache_path)['state_dict'])
        optimizer.load_state_dict(safe_torch_load(cache_path)['optimizer'])
        if safe_torch_load(cache_path)['scheduler_state'] is not None :
            scheduler.load_state_dict(safe_torch_load(cache_path)['scheduler_state'])
    model = model.to(DEVICE)
    if his_learner is not None:
        his_learner = his_learner.to(DEVICE)
    min_val_loss =  np.inf
    if args.cont > 0:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    wait = 0
    train_loss_list = []
    val_loss_list = []  
  
    print_log("------------------------------traffic predict training begin------------------------------", log=log)
    for epoch in range(args.cont + 1, max_epochs):
        train_time_s = time.time()

        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion,    clip_grad, epoch, mixup_num, log=log,w=w, device = DEVICE , shift_weight =None )
        train_time_e = time.time()
        

        val_loss  = eval_model(model, valset_loader, criterion,  epoch, scheduler=scheduler,  shift_weight = None )
        val_time_e = time.time()
        

        train_loss_list.append(train_loss)

        val_loss_list.append(val_loss)

        if (epoch) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch,
                " \tTrain Loss = %.5f" % train_loss, 
                "Val Loss = %.5f" % val_loss,

                "LR:%.5f" % optimizer.param_groups[0]['lr'],
                "mixup_num:%d" % mixup_num,
                "Train_time for one epoch:",
                  (train_time_e - train_time_s),
                " \tVal_time for one epoch:",
                (val_time_e - train_time_e),
                " \tscheduler loss:",
                (scheduler.best),
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
            save_checkpoint(model, optimizer, epoch, min_var_loss= min_val_loss,log = log)
        else:
            wait += 1
            if wait >= early_stop:
                break
        if args.tensorboard:
            w.add_scalars('loss_per_epoch',
                                    {'train_loss':train_loss,
                                        'valid_loss': val_loss}, epoch)
            w.add_scalar('LR',  optimizer.param_groups[0]['lr'], epoch)

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch}\n"
    out_str += f"Best at epoch {best_epoch}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch - 1]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch - 1]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)
    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader,   save_dir, epoch = 0, log=None, filter_threshold=0.005, output_dim=1):
    model.to(DEVICE)
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader,  epoch)
    end = time.time()
    outputs = {'pred': y_pred, 'true': y_true}
    filename = \
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + '_' \
                + 'predictions.npz'
    print("saving to:",os.path.join(save_dir, filename))
    np.savez_compressed(os.path.join(save_dir, filename), **outputs)
    
    # Determine output_dim from the last dimension
    assert y_true.shape == y_pred.shape, \
        f"形状不一致! y_true: {y_true.shape}, y_pred: {y_pred.shape}"
    
    # 当维度为3时，unsqueeze为4维
    if y_true.ndim == 3:
        y_true = np.expand_dims(y_true, axis=-1)
        y_pred = np.expand_dims(y_pred, axis=-1)
        print_log(f"扩展维度: y_true.shape -> {y_true.shape}, y_pred.shape -> {y_pred.shape}", log=log)

    print_log(f"y_true.shape {y_true.shape}, y_pred.shape {y_pred.shape}, y_true_step {y_true.shape[1]}, y_pred_step {y_pred.shape[1]}, output_dim {output_dim}", log=log)
    
    # 检查output_dim是否与配置一致
    assert y_true.shape[-1] == output_dim, \
        f"output_dim不一致! 数据最后一维: {y_true.shape[-1]}, 配置output_dim: {output_dim}"
    
    out_steps = y_pred.shape[1]

    
    # Calculate metrics for each output dimension
    for j in range(output_dim):
        if output_dim > 1:
            print_log(f"\n========== Output Dimension {j} (Feature {j}) ==========", log=log)
        # Average - cumulative average metrics from step 1 to step i
        print_log("\n--- Average (Cumulative from Step 1 to i) ---", log=log)
        out_str = ""
        for i in range(out_steps):
            # Calculate metrics from step 1 to step i+1
            rmse, mae, mape, smape, wape, mape_fil1x, mape_fil10x, mape_fil100x, mape_fil1000x, mape_fil10000x = ALL_METRICS(
                y_true[:, :i+1, ..., j], y_pred[:, :i+1, ..., j], filter_threshold=filter_threshold
            )
            out_str += "Step 1~%d RMSE = %.5f, MAE = %.5f, MAPE = %.5f, sMAPE = %.5f, WAPE = %.5f, MAPE_fil(>%.4f) = %.5f, MAPE_fil(>%.3f) = %.5f, MAPE_fil(>%.2f) = %.5f, MAPE_fil(>%.1f) = %.5f\n" % (
                i + 1, rmse, mae, mape, smape, wape, 
                filter_threshold, mape_fil1x, 
                filter_threshold*10, mape_fil10x, 
                filter_threshold*100, mape_fil100x, 
                filter_threshold*1000, mape_fil1000x, 
            )
        print_log(out_str, log=log, end="")
    
    # If multiple output dimensions, also print average across all dimensions
    if output_dim > 1:
        print_log("\n========== Average Across All Dimensions ==========", log=log)
        # Average
        print_log("\n--- Average (Cumulative from Step 1 to i) ---", log=log)
        out_str = ""
        for i in range(out_steps):
            rmse, mae, mape, smape, wape, mape_fil1x, mape_fil10x, mape_fil100x, mape_fil1000x, mape_fil10000x = ALL_METRICS(
                y_true[:, :i+1, :], y_pred[:, :i+1, :], filter_threshold=filter_threshold
            )
            out_str += "Step 1~%d RMSE = %.5f, MAE = %.5f, MAPE = %.5f, sMAPE = %.5f, WAPE = %.5f, MAPE_fil(>%.4f) = %.5f, MAPE_fil(>%.3f) = %.5f, MAPE_fil(>%.2f) = %.5f, MAPE_fil(>%.1f) = %.5f\n" % (
                i + 1, rmse, mae, mape, smape, wape, 
                filter_threshold, mape_fil1x, 
                filter_threshold*10, mape_fil10x, 
                filter_threshold*100, mape_fil100x, 
                filter_threshold*1000, mape_fil1000x, 
            )
        print_log(out_str, log=log, end="")

@torch.no_grad()
def test_simple(model, testset_loader,  epoch = 0, log=None, shift_weight = None, output_dim=1):
    model = model.to(DEVICE)
    model.eval()
    print_log("--------- Test ---------", log=log)

    y_true, y_pred = predict(model, testset_loader,   epoch )
    end = time.time()

    # Determine output_dim from the last dimension
    assert y_true.shape == y_pred.shape, \
        f"形状不一致! y_true: {y_true.shape}, y_pred: {y_pred.shape}"
    
    # 当维度为3时，unsqueeze为4维
    if y_true.ndim == 3:
        y_true = np.expand_dims(y_true, axis=-1)
        y_pred = np.expand_dims(y_pred, axis=-1)
        print_log(f"扩展维度: y_true.shape -> {y_true.shape}, y_pred.shape -> {y_pred.shape}", log=log)
    
    print_log(f"y_true.shape {y_true.shape}, y_pred.shape {y_pred.shape}, y_true_step {y_true.shape[1]}, y_pred_step {y_pred.shape[1]}, output_dim {output_dim}", log=log)
    
    # 检查output_dim是否与配置一致
    assert y_true.shape[-1] == output_dim, \
        f"output_dim不一致! 数据最后一维: {y_true.shape[-1]}, 配置output_dim: {output_dim}"

    out_steps = y_pred.shape[1]
    
    # Calculate metrics for each output dimension
    for j in range(output_dim):
        if output_dim > 1:
            print_log(f"\n========== Output Dimension {j} (Feature {j}) ==========", log=log)
        
        # Mode 1: Single - metrics for each individual step
        print_log("\n--- Mode: Single (Individual Steps) ---", log=log)
        out_str = ""
        for i in range(out_steps):
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, ..., j], y_pred[:, i, ..., j])
            out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
                i + 1,
                rmse,
                mae,
                mape,
            )
        print_log(out_str, log=log, end="")
        
        # Mode 2: Average - cumulative average metrics from step 1 to step i
        print_log("\n--- Mode: Average (Cumulative from Step 1 to i) ---", log=log)
        out_str = ""
        for i in range(out_steps):
            # Calculate metrics from step 1 to step i+1
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, :i+1, ..., j], y_pred[:, :i+1, ..., j])
            out_str += "Step 1~%d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
                i + 1,
                rmse,
                mae,
                mape,
            )
        print_log(out_str, log=log, end="")
    
    # If multiple output dimensions, also print average across all dimensions
    if output_dim > 1:
        print_log("\n========== Average Across All Dimensions ==========", log=log)
        
        # Mode 1: Single
        print_log("\n--- Mode: Single (Individual Steps) ---", log=log)
        out_str = ""
        for i in range(out_steps):
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
            out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
                i + 1,
                rmse,
                mae,
                mape,
            )
        print_log(out_str, log=log, end="")
        
        # Mode 2: Average
        print_log("\n--- Mode: Average (Cumulative from Step 1 to i) ---", log=log)
        out_str = ""
        for i in range(out_steps):
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, :i+1, :], y_pred[:, :i+1, :])
            out_str += "Step 1~%d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
                i + 1,
                rmse,
                mae,
                mape,
            )
        print_log(out_str, log=log, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="HZMETRO") 
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-t", "--tensorboard", action='store_true')
    parser.add_argument("-e", "--exp_id", type=int, default=1) 
    parser.add_argument("-c", "--comment", type=str, default='train')
    parser.add_argument("-his", "--his_dim", type=int, default=7)
    parser.add_argument("-m", "--mode", type=str, default='train')
    parser.add_argument("-cont", "--cont", type=int, default=0)
    parser.add_argument("-s", "--speed", type=bool, default=False)
    parser.add_argument("-full_rate", "--full_rate", type=float, default=1.0)
    
    # Solver
    parser.add_argument("--solver", type=str, default=None, help="ODE solver: euler, rk4, dopri5")
    parser.add_argument("--atol", type=float, default=None, help="Absolute tolerance for ODE solver")
    parser.add_argument("--rtol", type=float, default=None, help="Relative tolerance for ODE solver")
    parser.add_argument("--hid_dim", type=int, default=None, help="CDE hidden dimension")
    parser.add_argument("--hid_hid_dim", type=int, default=None, help="CDE internal hidden dimension")
    
    # Mixup
    parser.add_argument("--mixup_num", type=int, default=None, help="Mixup interval m (every m batches)")
    parser.add_argument("--mix_alpha", type=float, default=None, help="Mixup alpha parameter for Beta distribution")
    
    # HeteroGCNs
    parser.add_argument("--g_eps", type=int, default=None, help="Geographic mask threshold (Mgeo): distance <= g_eps")
    parser.add_argument("--s_eps", type=int, default=None, help="Semantic mask threshold (Msem): top-k neighbors")
    
    # Gaussian kernel
    parser.add_argument("--weight_adj_epsilon", type=float, default=None, help="Gaussian kernel threshold for adjacency matrix")
    
    # Loss
    parser.add_argument("--huber_delta", type=float, default=None, help="Huber loss delta parameter")
    parser.add_argument("--set_loss", type=str, default=None, help="Loss function: huber, l1, masked_mae")
    
    # train
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum training epochs")
    parser.add_argument("--early_stop", type=int, default=None, help="Early stopping patience")
    
    args = parser.parse_args()
    seed =2044 # set random seed here
    init_seed(seed)
    set_cpu_num(40)
    
    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}/"
    model_name = STFACH.__name__
    if args.exp_id is None:
        args.exp_id = int(random.SystemRandom().random() * 100000)

    save_name = f"{str(args.exp_id)}_{dataset}_{model_name}_{str(args.comment)}"
    path = '../runs'
    log_dir = os.path.join(path, dataset, save_name)

    if (os.path.exists(log_dir)):
        print('has model save path')
    else:
        os.makedirs(log_dir)

    with open(f"{model_name}.yaml", "r",encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]
    
    # ============ args for config ============
    # Solver
    if args.solver is not None:
        cfg["model_args"]["solver"] = args.solver
        print(f"Override solver: {args.solver}")
    if args.atol is not None:
        cfg["model_args"]["atol"] = args.atol
        print(f"Override atol: {args.atol}")
    if args.rtol is not None:
        cfg["model_args"]["rtol"] = args.rtol
        print(f"Override rtol: {args.rtol}")
    if args.hid_dim is not None:
        cfg["model_args"]["hid_dim"] = args.hid_dim
        print(f"Override hid_dim: {args.hid_dim}")
    if args.hid_hid_dim is not None:
        cfg["model_args"]["hid_hid_dim"] = args.hid_hid_dim
        print(f"Override hid_hid_dim: {args.hid_hid_dim}")
    
    # HeteroGCNs
    if args.g_eps is not None:
        cfg["model_args"]["g_eps"] = args.g_eps
        print(f"Override g_eps: {args.g_eps}")
    if args.s_eps is not None:
        cfg["model_args"]["s_eps"] = args.s_eps
        print(f"Override s_eps: {args.s_eps}")
    
    # Loss
    if args.set_loss is not None:
        cfg["set_loss"] = args.set_loss
        print(f"Override set_loss: {args.set_loss}")
    if args.huber_delta is not None:
        cfg["huber_delta"] = args.huber_delta
        print(f"Override huber_delta: {args.huber_delta}")
    
    # train
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
        print(f"Override max_epochs: {args.max_epochs}")
    if args.early_stop is not None:
        cfg["early_stop"] = args.early_stop
        print(f"Override early_stop: {args.early_stop}")

 # -------------------------------- tensorboard -------------------------------- #
    if args.tensorboard:
        tensorboard_dir = os.path.join(data_path, str('vis'),str(args.exp_id)+"_"+time.strftime("%m-%d-%Hh%Mm")).replace("\\", "/")
        if not (os.path.exists(tensorboard_dir)):
            os.makedirs(tensorboard_dir)
        args.t_dir = tensorboard_dir
        w   = SummaryWriter(log_dir=tensorboard_dir)
        print(tensorboard_dir)
    else:
        w = None

   
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%m_%d(%H-%M-%S)")
    log = os.path.join(log_dir, f"{args.mode}-{model_name}-{dataset}-{now}-cmt{args.comment}-e{args.exp_id}-c{args.cont}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    his_log = os.path.join(log_dir, f"his_learner-{args.mode}-{dataset}-{now}-cmt{args.comment}-e{args.exp_id}-c{args.cont}.log")
    his_log = open(his_log, "a")
    his_log.seek(0)
    his_log.truncate()


    # ------------------------------- load dataset ------------------------------- #
    print_log(dataset, log=log)
    
    # Determine data type: 'grid' or 'point' based on config or file existence
    data_type = cfg.get("data_type", "point")  # Default to 'point' for backward compatibility
    if data_type == "point":
        # Check if .dyna file exists for point data
        if not os.path.exists(os.path.join(data_path, f"{dataset}.dyna")):
            # If .dyna doesn't exist but .grid exists, use grid data
            if os.path.exists(os.path.join(data_path, f"{dataset}.grid")):
                data_type = "grid"
                print_log(f"Auto-detected grid data type for {dataset}", log=log)
    
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
        adj_mx,
        adj_semx
    ) = generate_data(
        data_dir=data_path,
        dataset=dataset,
        data_col=cfg.get("data_col"),
        output_dim=cfg["model_args"]["output_channels"],
        data_type=data_type,
        batch_size=cfg.get("batch_size", 64),
        load_dtw=cfg.get("load_dtw", False),
        load_external=True,
        train_rate=cfg.get("train_size", 0.6),
        eval_rate=cfg.get("val_size", 0.2),
        full_rate=args.full_rate,
        use_row_column=False,
        time_intervals=cfg.get("time_intervals", 300),
        log=log,
        log_dir=log_dir,
        in_steps= cfg.get("in_steps", 12),
        out_steps=cfg.get("out_steps", 12),
        mode=args.mode,
        weight_adj_epsilon=args.weight_adj_epsilon if args.weight_adj_epsilon is not None else 0.1
    )
    
    print_log(f"Data type: {data_type}", log=log)

    
    
    print_log(log=log)
     # -------------------------------- load model -------------------------------- #

    model = STFACH(**cfg["model_args"], adj_mx = adj_mx, adj_semx = adj_semx)
    his_learner = HisPattern(dim = args.his_dim)


    # --------------------------- set model saving path -------------------------- #
 
    save = os.path.join(log_dir, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #
    set_loss = cfg.get("set_loss", 'huber')
    
    # Standard PyTorch losses
    if set_loss == 'huber':
        delta = cfg.get("huber_delta", 2.5)
        criterion = nn.HuberLoss(delta=delta)
        print_log(f"Using HuberLoss with delta={delta}", log=log)
    elif set_loss == 'l1':
        criterion = nn.L1Loss()
        print_log("Using L1Loss", log=log)
    elif set_loss == 'mse':
        criterion = nn.MSELoss()
        print_log("Using MSELoss", log=log)
    elif set_loss == 'masked_mae':
        criterion = MaskedMAELoss()
        print_log("Using MaskedMAELoss", log=log)
    
    # Custom losses for sparse data
    elif set_loss == 'logcosh':
        criterion = get_loss_function('logcosh')
        print_log("Using LogCoshLoss (sensitive to both small and large errors)", log=log)
    elif set_loss == 'adaptive_huber':
        base_delta = cfg.get("base_delta", 1.0)
        scale_factor = cfg.get("scale_factor", 0.1)
        criterion = get_loss_function('adaptive_huber', base_delta=base_delta, scale_factor=scale_factor)
        print_log(f"Using AdaptiveHuberLoss (base_delta={base_delta}, scale_factor={scale_factor})", log=log)
    elif set_loss == 'weighted_mse':
        weight_type = cfg.get("weight_type", 'inverse')
        epsilon = cfg.get("loss_epsilon", 1.0)
        criterion = get_loss_function('weighted_mse', weight_type=weight_type, epsilon=epsilon)
        print_log(f"Using WeightedMSELoss (weight_type={weight_type}, epsilon={epsilon})", log=log)
    elif set_loss == 'balanced_l1':
        threshold = cfg.get("balance_threshold", 10.0)
        alpha = cfg.get("balance_alpha", 2.0)
        criterion = get_loss_function('balanced_l1', threshold=threshold, alpha=alpha)
        print_log(f"Using BalancedL1Loss (threshold={threshold}, alpha={alpha})", log=log)
    elif set_loss == 'combined':
        mae_weight = cfg.get("mae_weight", 1.0)
        mape_weight = cfg.get("mape_weight", 0.5)
        mse_weight = cfg.get("mse_weight", 0.5)
        epsilon = cfg.get("loss_epsilon", 1.0)
        criterion = get_loss_function('combined', mae_weight=mae_weight, mape_weight=mape_weight, 
                                     mse_weight=mse_weight, epsilon=epsilon)
        print_log(f"Using CombinedLoss (MAE={mae_weight}, MAPE={mape_weight}, MSE={mse_weight})", log=log)
    elif set_loss == 'focal_mse':
        gamma = cfg.get("focal_gamma", 2.0)
        criterion = get_loss_function('focal_mse', gamma=gamma)
        print_log(f"Using FocalMSELoss (gamma={gamma})", log=log)
    elif set_loss == 'rmsle':
        epsilon = cfg.get("loss_epsilon", 1.0)
        criterion = get_loss_function('rmsle', epsilon=epsilon)
        print_log(f"Using RMSLELoss (epsilon={epsilon})", log=log)
    elif set_loss == 'mape':
        epsilon = cfg.get("loss_epsilon", 1.0)
        criterion = get_loss_function('mape', epsilon=epsilon)
        print_log(f"Using MAPELoss (epsilon={epsilon})", log=log)
    elif set_loss == 'quantile':
        quantile = cfg.get("quantile", 0.5)
        criterion = get_loss_function('quantile', quantile=quantile)
        print_log(f"Using QuantileLoss (quantile={quantile})", log=log)
    else:
        criterion = nn.HuberLoss(delta=2.5)
        print_log(f"Unknown loss '{set_loss}', using default HuberLoss", log=log)  


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
        betas=(0.9, 0.999)
    )

    his_optimizer = torch.optim.Adam(his_learner.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.35,min_lr=0.000001,patience=5,  threshold_mode = 'rel',cooldown=0,threshold=0.0001) 
    
    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log("---------", model, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)
    

    # --------------------------- Curriculum Learning --------------------------- #
    iter_count = 0
    if cfg["use_cl"]:
        print_log(f"Applying Curriculum Learning", log=log)
        target_length = 0
        print_log(f"CL target length = {target_length}", log=log)

    else :
        target_length = cfg["out_steps"]

 
 
    if args.cont > 0 :
        cache_path  =os.path.join(log_dir,f"model_ckpt_{args.cont}.pth")
        print_log(f"loaded pretrain epoch of {args.cont} at {cache_path}", log = log)
    else:
        cache_path = None


    #### get mixup sample rate among data ####
    all_begin = time.time()
    mix_up = True
 
    sample_use_time = time.time() - all_begin
    
    # Mixup
    mixup_num = args.mixup_num if args.mixup_num is not None else 5
    mix_alpha = args.mix_alpha if args.mix_alpha is not None else 2.0
    
    input_dim = cfg["model_args"]["input_dim"]

   
    if args.mode == 'train':
        model = train(
            model,
            trainset_loader,
            valset_loader,
            optimizer,
            scheduler,
            criterion,
            adj_mx,
            clip_grad=cfg.get("clip_grad"),
            max_epochs=cfg.get("max_epochs", 200),
            early_stop=cfg.get("early_stop", 10),
            verbose=1,
            log=log,
            save=save,
            cache_path = cache_path,
            w=w ,#tensorboard
            his_opt = his_optimizer,
            his_learner = his_learner,
            mixup_num=mixup_num,
            mix_alpha=mix_alpha
        )
        print_log(f"Saved Model: {save}", log=log)
        test_model(model, testset_loader,  save_dir = log_dir, log=log, filter_threshold = cfg.get("filter_threshold", 0.001), output_dim=cfg["model_args"]["output_channels"])

    elif args.mode == 'test':
        print_log("--------", model, "---------", log=log)
        model.load_state_dict(safe_torch_load(cache_path)['state_dict'])
        test_model(model, testset_loader,  save_dir = log_dir, log=log, filter_threshold = cfg.get("filter_threshold", 0.001), output_dim=cfg["model_args"]["output_channels"])
    
    
    elif args.mode == 'his':
        print_log("========== MODE: PRETRAIN HISTORY WEIGHTS ==========", log=log)
        print_log(f"Dataset: {dataset}", log=log)
        print_log(f"History learner dimension: {args.his_dim}", log=log)
        print_log(log=log)
        his_learner = his_learner.to(DEVICE)
        # Pretrain history weights only
        his_epochs = 200
        min_his_var = np.inf
        his_criterion = nn.HuberLoss(delta=2.5)
        best_his_state = None
        
        make_dir(log_dir + '/his_learn')
        make_dir('../finetune_his')
        saved_his_path = log_dir + '/his_learn/' + args.dataset + '_final_best_hisweight.pth'
        
        print_log("------------------------------his training begin------------------------------", log=log)
        for epoch in range(0, his_epochs):
            his_loss_list = []
            
            # Training for his learner
            his_learner.train()
            for batch_idx, batch in enumerate(trainset_loader):
                batch = tuple(b.to(DEVICE) for b in batch)
                x_batch, y_batch,  *coeff_batch = batch
                shifts = x_batch[..., -args.his_dim:]
                weighted_shifts = his_learner(shifts)
                weighted_shifts = SCALER.inverse_transform(weighted_shifts)
                x_batch = x_batch.mean(axis=-2)
                y_batch = y_batch.mean(axis=-2)
                
                history_loss = his_criterion(weighted_shifts, y_batch)
                his_loss_list.append(history_loss.item())
                
                his_optimizer.zero_grad()
                history_loss.backward()
                his_optimizer.step()
            
            his_train_loss = np.mean(his_loss_list)
            
            # Validating for his learner
            his_loss_list = []
            his_learner.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(valset_loader):
                    batch = tuple(b.to(DEVICE) for b in batch)
                    x_batch, y_batch,  *coeff_batch = batch
                    shifts = x_batch[..., -args.his_dim:]
                    weighted_shifts = his_learner(shifts)
                    x_batch = x_batch.mean(axis=-2)
                    y_batch = y_batch.mean(axis=-2)
                    weighted_shifts = SCALER.inverse_transform(weighted_shifts)
                    
                    history_loss = his_criterion(weighted_shifts, y_batch)
                    his_loss_list.append(history_loss.item())
                
                his_val_loss = np.mean(his_loss_list)
            
            if (epoch) % 1 == 0:
                print_log(
                    datetime.datetime.now(),
                    "Epoch",
                    epoch,
                    " \tHistrain Loss = %.5f" % his_train_loss,
                    "Hisval Loss = %.5f" % his_val_loss,
                    log=log,
                )
            
            if his_val_loss < min_his_var:
                min_his_var = his_val_loss
                best_his_state = his_learner.state_dict()
                print_log(f"Saving best his_learner at epoch {epoch}", log=log)
        
        # Save best history weights
        shift_weight = torch.sigmoid(best_his_state['weight_shift'])
        np.save(f"../finetune_his/{args.dataset}_final_best_hisweight.npy", shift_weight.cpu().detach().numpy())
        torch.save({'state_dict': best_his_state}, saved_his_path)
        
        print_log(f"Saved best his_learner to: {saved_his_path}", log=log)
        print_log(f"Saved weight array to: ../finetune_his/{args.dataset}_final_best_hisweight.npy", log=log)
        print_log("========== HISTORY WEIGHT PRETRAINING COMPLETED ==========", log=log)
    else:
        print_log(f"Unknown mode: {args.mode}", log=log)
        print_log("Available modes: train, test, cal, his, ans, vis", log=log)

    log.close()
