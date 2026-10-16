import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    """
    Differentiable Correlation coefficient to measure trend similarity.
    Supports 2D [B, N] and 3D [B, T, N] inputs.
    """
    if isinstance(pred, torch.Tensor):
        if pred.ndim == 2:
            pred = pred.unsqueeze(1)
            true = true.unsqueeze(1)
        
        B, T, C = pred.shape
        pred_flat = pred.reshape(-1, C)
        true_flat = true.reshape(-1, C)
        
        # Pearson correlation per channel
        pred_mean = torch.mean(pred_flat, dim=0)
        true_mean = torch.mean(true_flat, dim=0)
        
        u = torch.sum((pred_flat - pred_mean) * (true_flat - true_mean), dim=0)
        d = torch.sqrt(torch.sum((pred_flat - pred_mean)**2, dim=0) * torch.sum((true_flat - true_mean)**2, dim=0) + 1e-8)
        
        return torch.mean(u / d)
    else:
        if pred.ndim == 2:
            pred = np.expand_dims(pred, axis=1)
            true = np.expand_dims(true, axis=1)
        B, T, C = pred.shape
        pred_flat = pred.reshape(-1, C)
        true_flat = true.reshape(-1, C)
        u = ((true_flat - true_flat.mean(0)) * (pred_flat - pred_flat.mean(0))).sum(0)
        d = np.sqrt(((true_flat - true_flat.mean(0))**2).sum(0) * ((pred_flat - pred_flat.mean(0))**2).sum(0))
        return (u / (d + 1e-8)).mean()

def TDC(pred, true):
    """
    Temporal Dynamic Correlation: 
    Measures how well the change rate (delta) of the prediction matches the ground truth.
    """
    if isinstance(pred, torch.Tensor):
        if pred.ndim == 2 or pred.shape[1] < 2:
            return torch.tensor(0.0).to(pred.device)
        # Calculate first-order difference
        pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
        true_delta = true[:, 1:, :] - true[:, :-1, :]
        return CORR(pred_delta, true_delta)
    else:
        if pred.ndim == 2 or pred.shape[1] < 2:
            return 0.0
        pred_delta = np.diff(pred, axis=1)
        true_delta = np.diff(true, axis=1)
        return CORR(pred_delta, true_delta)

def QuantileLoss(pred, true, q=0.5):
    """
    Pinball loss for quantile evaluation.
    """
    e = true - pred
    return torch.mean(torch.max(q * e, (q - 1) * e))

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    return torch.mean((pred - true) ** 2)

def metric(pred, true):
    mse = MSE(pred, true).item()
    mae = MAE(pred, true).item()
    corr = CORR(pred, true)
    tdc = TDC(pred, true)
    p50 = QuantileLoss(pred, true, 0.5).item()
    p90 = QuantileLoss(pred, true, 0.9).item()
    
    return mse, mae, corr, tdc, p50, p90