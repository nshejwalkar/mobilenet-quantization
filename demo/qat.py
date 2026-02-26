"""
Quantization-Aware Training (QAT) experiments module.

This module implements five QAT ablation studies with different layer-selective strategies.
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluate, print_size_of_model, get_parameter_counts


def run_full_qat(baseline_model, trainloader, testloader, device, epochs=10):
    """
    Run QAT with all layers trainable (Experiment 1).
    
    Args:
        baseline_model: Baseline FP32 model
        trainloader: DataLoader for training
        testloader: DataLoader for testing
        device: Device to train on
        epochs: Number of QAT epochs
        
    Returns:
        tuple: (int8_model, accuracy, model_size, trainable_params, total_params)
    """
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    
    exp1_model = copy.deepcopy(baseline_model)
    exp1_model.to(device)
    exp1_model.train()
    exp1_model.fuse_model()
    
    # QAT config
    exp1_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(exp1_model, inplace=True)
    total, trainable = get_parameter_counts(exp1_model)
    
    optimizer = optim.SGD(exp1_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f'Starting QAT training for {epochs} epochs')
    for epoch in range(epochs):
        start = time.time()
        exp1_model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = exp1_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        epoch_time = time.time() - start
        cur_acc = evaluate(exp1_model, testloader, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {cur_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    exp1_model.eval().to('cpu')
    int8_exp1 = torch.quantization.convert(exp1_model, inplace=False)
    acc_exp1 = evaluate(int8_exp1, testloader, device='cpu')
    size_exp1 = print_size_of_model(int8_exp1)
    
    return int8_exp1, acc_exp1, size_exp1, trainable, total


def run_classifier_qat(baseline_model, trainloader, testloader, device, epochs=10):
    """
    Run QAT with only classifier trainable (Experiment 2).
    
    Args:
        baseline_model: Baseline FP32 model
        trainloader: DataLoader for training
        testloader: DataLoader for testing
        device: Device to train on
        epochs: Number of QAT epochs
        
    Returns:
        tuple: (int8_model, accuracy, model_size, trainable_params, total_params)
    """
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    
    exp2_model = copy.deepcopy(baseline_model)
    exp2_model.to(device)
    exp2_model.train()
    exp2_model.fuse_model()
    
    exp2_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(exp2_model, inplace=True)
    
    # freezing logic
    for name, param in exp2_model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    total, trainable = get_parameter_counts(exp2_model)
    print(f'Retraining {trainable:,} / {total:,} parameters ({trainable/total:.1%})')
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, exp2_model.parameters()), 
                         lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        start = time.time()
        exp2_model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = exp2_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        cur_acc = evaluate(exp2_model, testloader, device)
        epoch_time = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {cur_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    exp2_model.eval().to('cpu')
    int8_exp2 = torch.quantization.convert(exp2_model, inplace=False)
    acc_exp2 = evaluate(int8_exp2, testloader, device='cpu')
    size_exp2 = print_size_of_model(int8_exp2)
    
    return int8_exp2, acc_exp2, size_exp2, trainable, total


def run_depthwise_qat(baseline_model, trainloader, testloader, device, epochs=10):
    """
    Run QAT with depthwise convolutions and classifier trainable (Experiment 3).
    
    Args:
        baseline_model: Baseline FP32 model
        trainloader: DataLoader for training
        testloader: DataLoader for testing
        device: Device to train on
        epochs: Number of QAT epochs
        
    Returns:
        tuple: (int8_model, accuracy, model_size, trainable_params, total_params)
    """
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    
    exp3_model = copy.deepcopy(baseline_model)
    exp3_model.to(device)
    exp3_model.train()
    exp3_model.fuse_model()
    
    exp3_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(exp3_model, inplace=True)
    
    # freezing logic
    for name, module in exp3_model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.groups == module.in_channels:
            for p in module.parameters():
                p.requires_grad = True
        elif "classifier" in name:
            for p in module.parameters():
                p.requires_grad = True
        else:
            for p in module.parameters():
                p.requires_grad = False
    
    total, trainable = get_parameter_counts(exp3_model)
    print(f'Retraining {trainable:,} / {total:,} parameters ({trainable/total:.1%})')
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, exp3_model.parameters()), 
                         lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        start = time.time()
        exp3_model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = exp3_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        cur_acc = evaluate(exp3_model, testloader, device)
        epoch_time = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {cur_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    exp3_model.eval().to('cpu')
    int8_exp3 = torch.quantization.convert(exp3_model, inplace=False)
    acc_exp3 = evaluate(int8_exp3, testloader, device='cpu')
    size_exp3 = print_size_of_model(int8_exp3)
    
    return int8_exp3, acc_exp3, size_exp3, trainable, total


def run_depthwise_1x1_qat(baseline_model, trainloader, testloader, device, epochs=10):
    """
    Run QAT with depthwise + 1x1 projection convolutions trainable (Experiment 4).
    
    Args:
        baseline_model: Baseline FP32 model
        trainloader: DataLoader for training
        testloader: DataLoader for testing
        device: Device to train on
        epochs: Number of QAT epochs
        
    Returns:
        tuple: (int8_model, accuracy, model_size, trainable_params, total_params)
    """
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    
    exp4_model = copy.deepcopy(baseline_model)
    exp4_model.to(device)
    exp4_model.train()
    exp4_model.fuse_model()
    
    exp4_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(exp4_model, inplace=True)
    
    for name, param in exp4_model.named_parameters():
        param.requires_grad = False
        # classifier always trainable
        if "classifier" in name:
            param.requires_grad = True
            continue
        # depthwise
        if "features.1.conv.0.0.weight" in name:      # depthwise in block 1
            param.requires_grad = True
        if ".conv.1.0.weight" in name:                # depthwise in blocks 2+
            param.requires_grad = True
        # projection 1×1 (last conv in block)
        if ".conv.1.weight" in name and "features.1." in name:  # proj in block 1
            param.requires_grad = True
        if ".conv.2.weight" in name:                            # proj in blocks 2+
            param.requires_grad = True
    
    total, trainable = get_parameter_counts(exp4_model)
    print(f'Retraining {trainable:,} / {total:,} parameters ({trainable/total:.1%})')
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, exp4_model.parameters()), 
                         lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        start = time.time()
        exp4_model.train()
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = exp4_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        cur_acc = evaluate(exp4_model, testloader, device)
        epoch_time = time.time() - start
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {cur_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    exp4_model.eval().to('cpu')
    int8_exp4 = torch.quantization.convert(exp4_model, inplace=False)
    acc_exp4 = evaluate(int8_exp4, testloader, device='cpu')
    size_exp4 = print_size_of_model(int8_exp4)
    
    return int8_exp4, acc_exp4, size_exp4, trainable, total


def run_depthwise_1x1_asymmetric_qat(baseline_model, trainloader, testloader, device, epochs=10):
    """
    Run QAT with depthwise + 1x1 trainable and asymmetric quantization (Experiment 5).
    
    Args:
        baseline_model: Baseline FP32 model
        trainloader: DataLoader for training
        testloader: DataLoader for testing
        device: Device to train on
        epochs: Number of QAT epochs
        
    Returns:
        tuple: (int8_model, accuracy, model_size, trainable_params, total_params)
    """
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    
    exp5_model = copy.deepcopy(baseline_model)
    exp5_model.to(device)
    exp5_model.train()
    exp5_model.fuse_model()
    
    exp5_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(exp5_model, inplace=True)
    
    for name, param in exp5_model.named_parameters():
        param.requires_grad = False
        # classifier always trainable
        if "classifier" in name:
            param.requires_grad = True
            continue
        # depthwise
        if "features.1.conv.0.0.weight" in name:      # depthwise in block 1
            param.requires_grad = True
        if ".conv.1.0.weight" in name:                # depthwise in blocks 2+
            param.requires_grad = True
        # projection 1×1 (last conv in block)
        if ".conv.1.weight" in name and "features.1." in name:  # proj in block 1
            param.requires_grad = True
        if ".conv.2.weight" in name:                            # proj in blocks 2+
            param.requires_grad = True
    
    total, trainable = get_parameter_counts(exp5_model)
    print(f'Retraining {trainable:,} / {total:,} parameters ({trainable/total:.1%})')
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, exp5_model.parameters()), 
                         lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        start = time.time()
        exp5_model.train()
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = exp5_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        cur_acc = evaluate(exp5_model, testloader, device)
        epoch_time = time.time() - start
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {cur_acc:.2f}% | Time: {epoch_time:.1f}s")
    
    exp5_model.eval().to('cpu')
    int8_exp5 = torch.quantization.convert(exp5_model, inplace=False)
    acc_exp5 = evaluate(int8_exp5, testloader, device='cpu')
    size_exp5 = print_size_of_model(int8_exp5)
    
    return int8_exp5, acc_exp5, size_exp5, trainable, total
