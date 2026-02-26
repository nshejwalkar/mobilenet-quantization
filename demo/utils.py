"""
Utility functions for MobileNetV2 quantization experiments.

This module contains helper functions for:
- Model evaluation and metrics
- Plotting and visualization
- Performance measurement
"""

import os
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def evaluate(model, loader, device):
    """
    Evaluate model accuracy on a given data loader.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader containing evaluation data
        device: Device to run evaluation on
        
    Returns:
        float: Accuracy percentage
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def print_size_of_model(model):
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    torch.save(model.state_dict(), 'temp.p')
    size_mb = os.path.getsize('temp.p')/1e6
    os.remove('temp.p')
    return size_mb


def get_parameter_counts(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def plot_saturation(int8_model, loader, device='cpu'):
    """
    Plot histogram of INT8 quantized values to visualize saturation.
    
    Args:
        int8_model: Quantized INT8 model
        loader: DataLoader for input data
        device: Device to run on
    """
    int8_model.eval()

    target_layer = int8_model.features[14].conv[0]

    int8_values = []

    def hook(module, input, output):
        int_repr = output.int_repr().flatten().float().numpy()
        int8_values.extend(int_repr)

    handle = target_layer.register_forward_hook(hook)

    inputs, _ = next(iter(loader))
    int8_model(inputs.to(device))
    handle.remove()

    plt.figure(figsize=(10, 5))
    plt.hist(int8_values, bins=128, range=(0, 127), color='purple', alpha=0.7)
    plt.yscale('log')
    plt.title("Saturation Histogram (INT8 Values in Depthwise Layer)")
    plt.xlabel("Quantized Value (Integer)")
    plt.ylabel("Count (Log Scale)")

    plt.axvline(x=0, color='r', linestyle='--', label='Min Limit')
    plt.axvline(x=127, color='r', linestyle='--', label='Max Limit')
    plt.legend()

    plt.tight_layout()


def plot_layer_error(fp32_model, int8_model, loader, device='cpu'):
    """
    Plot layer-wise quantization error between FP32 and INT8 models.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized INT8 model
        loader: DataLoader for input data
        device: Device to run on
    """
    fp32_model.eval()
    fp32_model.to(device)
    int8_model.eval()
    int8_model.to(device)
    indices = [0, 3, 7, 10, 14, 18]

    fp32_outputs = {}
    int8_outputs = {}

    def get_hook(storage_dict, idx):
        def hook(module, inp, out):
            if hasattr(out, 'dequantize'):
                storage_dict[idx] = out.dequantize().detach().cpu()
            else:
                storage_dict[idx] = out.detach().cpu()
        return hook

    # register hooks
    handles = []
    for i in indices:
        # FP32 hooks
        h1 = fp32_model.features[i].register_forward_hook(get_hook(fp32_outputs, i))
        handles.append(h1)

        # INT8 hooks
        h2 = int8_model.features[i].register_forward_hook(get_hook(int8_outputs, i))
        handles.append(h2)

    img, _ = next(iter(loader))
    fp32_model(img.to(device))
    int8_model(img.to('cpu'))
    for h in handles: h.remove()
    mses = []
    layers = []

    for i in indices:
        f = fp32_outputs[i]
        q = int8_outputs[i]
        error_norm = torch.norm(f - q)
        signal_norm = torch.norm(f)
        if signal_norm > 0:
            rel_error = error_norm / signal_norm
        else:
            rel_error = 0.0
        mses.append(rel_error.item())
        layers.append(f"Layer {i}")

    plt.figure(figsize=(10, 5))
    plt.plot(layers, mses, marker='o', color='red', linestyle='--', linewidth=2)
    plt.xlabel("Network Depth")
    plt.ylabel("Relative Error (Noise-to-Signal Ratio)")
    plt.title("True Error Accumulation (Relative Drift)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_channel_variance(model, loader, device='cpu'):
    """
    Plot inter-channel activation variance for a target layer.
    
    Args:
        model: PyTorch model
        loader: DataLoader for input data
        device: Device to run on
    """
    model.eval()
    target_layer = model.features[14].conv[0]

    activation_ranges = []

    def hook(module, input, output):
        flat = output.detach().view(output.shape[0], output.shape[1], -1)
        batch_max = flat.max(dim=2)[0].mean(dim=0)
        batch_min = flat.min(dim=2)[0].mean(dim=0)

        rng = batch_max - batch_min
        activation_ranges.append(rng.cpu().numpy())

    handle = target_layer.register_forward_hook(hook)

    inputs, _ = next(iter(loader))
    model(inputs.to(device))

    handle.remove()

    ranges = activation_ranges[0]

    plt.figure(figsize=(10, 5))
    plt.plot(ranges, marker='o', linestyle='-', markersize=2, alpha=0.7, color='teal')
    plt.title(f"Inter-Channel Activation Variance (Layer: features.14)\nTarget: Depthwise Conv")
    plt.xlabel("Channel Index")
    plt.ylabel("Dynamic Range (Max - Min)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()


def compute_activation_drift(fp32_model, int8_model, testloader):
    """
    Compute activation drift between FP32 and INT8 models across all blocks.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized INT8 model
        testloader: DataLoader for test data
        
    Returns:
        list: Drift values for each block
    """
    fp32_model = fp32_model.eval().to('cpu')
    int8_model = int8_model.eval().to('cpu')

    inputs, _ = next(iter(testloader))
    inputs = inputs.to('cpu')

    fp32_outs = {}
    int8_outs = {}

    def make_hook(storage, name):
        def hook(module, inp, out):
            storage[name] = out.detach()
        return hook

    hooks_fp32, hooks_int8 = [], []

    for idx, block in enumerate(fp32_model.features):
        hooks_fp32.append(block.register_forward_hook(make_hook(fp32_outs, f"b{idx}")))
    for idx, block in enumerate(int8_model.features):
        hooks_int8.append(block.register_forward_hook(make_hook(int8_outs, f"b{idx}")))

    with torch.no_grad():
        fp32_model(inputs)
        int8_model(inputs)

    drifts = []
    num_blocks = len(fp32_outs)

    for i in range(num_blocks):
        a = fp32_outs[f"b{i}"]
        b = int8_outs[f"b{i}"]
        if b.is_quantized:
            b = b.dequantize()
        drift = torch.norm(a - b) / torch.norm(a)
        drifts.append(drift.item())

    plt.figure(figsize=(12, 5))
    plt.plot(drifts, marker='o', linewidth=2)
    plt.title("Activation Drift Per Block (FP32 vs Depthwise-QAT INT8)")
    plt.xlabel("MobileNetV2 Block Index")
    plt.ylabel("Normalized L2 Drift")
    plt.grid(True)
    plt.show()
    for h in hooks_fp32: h.remove()
    for h in hooks_int8: h.remove()

    return drifts


def measure_throughput(model, testloader, device='cpu', batch_size=256, reps=50):
    """
    Measure model throughput (images per second).
    
    Args:
        model: PyTorch model
        testloader: DataLoader for test data
        device: Device to run on
        batch_size: Batch size for throughput measurement
        reps: Number of repetitions
        
    Returns:
        float: Throughput in images per second
    """
    model.eval()
    model.to(device)
    data_iter = iter(testloader)
    inputs, _ = next(data_iter)
    inputs = inputs[:batch_size].to(device)
    for _ in range(10):
        _ = model(inputs)
    start = time.time()
    for _ in range(reps):
        _ = model(inputs)
    end = time.time()
    total_images = batch_size * reps
    throughput = total_images / (end - start)
    return throughput
