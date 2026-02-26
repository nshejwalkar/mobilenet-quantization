"""
Post-Training Quantization (PTQ) experiments module.

This module implements PTQ with per-tensor and per-channel quantization schemes.
"""

import torch
import torch.nn as nn
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2
from utils import evaluate, print_size_of_model, plot_channel_variance, plot_saturation


def run_ptq_per_tensor(checkpoint_path, calibloader, testloader):
    """
    Run PTQ with per-tensor symmetric quantization.
    
    Args:
        checkpoint_path: Path to baseline model checkpoint
        calibloader: DataLoader for calibration
        testloader: DataLoader for testing
        
    Returns:
        tuple: (int8_model, accuracy, model_size)
    """
    ptq_model = quantizable_mobilenet_v2(weights=None, quantize=False)
    ptq_model.classifier[1] = nn.Linear(ptq_model.last_channel, 10)
    ptq_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    ptq_model.to('cpu')
    ptq_model.eval()
    
    # fuse model to PTQ instance
    ptq_model.fuse_model()
    
    # prepping for PTQ
    per_tensor_qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_observer,
        weight=torch.quantization.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    )
    
    ptq_model.qconfig = per_tensor_qconfig
    torch.quantization.prepare(ptq_model, inplace=True)
    
    # calibrating data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(calibloader):
            if i >= 10:
                break
            ptq_model(inputs)
    
    # convert to INT8
    plot_channel_variance(ptq_model, testloader, device='cpu')
    int8_model = torch.quantization.convert(ptq_model, inplace=False)
    plot_saturation(int8_model, testloader, device='cpu')
    acc_int8 = evaluate(int8_model, testloader, device='cpu')
    size_int8 = print_size_of_model(int8_model)
    
    return int8_model, acc_int8, size_int8


def run_ptq_per_channel(checkpoint_path, calibloader, testloader):
    """
    Run PTQ with per-channel symmetric quantization.
    
    Args:
        checkpoint_path: Path to baseline model checkpoint
        calibloader: DataLoader for calibration
        testloader: DataLoader for testing
        
    Returns:
        tuple: (int8_model, accuracy, model_size)
    """
    ptq_model = quantizable_mobilenet_v2(weights=None, quantize=False)
    ptq_model.classifier[1] = nn.Linear(ptq_model.last_channel, 10)
    ptq_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    ptq_model.to('cpu')
    ptq_model.eval()
    
    # fuse model to PTQ instance
    ptq_model.fuse_model()
    
    # prepping for PTQ
    ptq_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    torch.quantization.prepare(ptq_model, inplace=True)
    
    # calibrating data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(calibloader):
            if i >= 10:
                break
            ptq_model(inputs)
    
    # convert to INT8
    int8_model = torch.quantization.convert(ptq_model, inplace=False)
    
    acc_int8 = evaluate(int8_model, testloader, device='cpu')
    size_int8 = print_size_of_model(int8_model)
    
    return int8_model, acc_int8, size_int8
