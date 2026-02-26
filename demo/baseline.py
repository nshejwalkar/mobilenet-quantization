"""
Baseline FP32 MobileNetV2 training module.

This module handles fine-tuning a pre-trained MobileNetV2 model on CIFAR-10.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def train_baseline_model(model, trainloader, testloader, device, num_epochs=60, save_dir='./results'):
    """
    Train baseline FP32 MobileNetV2 model on CIFAR-10.
    
    Args:
        model: MobileNetV2 model to train
        trainloader: DataLoader for training data
        testloader: DataLoader for testing data
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
        save_dir: Directory to save results
        
    Returns:
        float: Best accuracy achieved during training
    """
    os.makedirs(f'{save_dir}/baseline', exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
    
    writer = SummaryWriter(f'{save_dir}/runs/mobilenet_baseline_experiment')
    print(f'Training MobileNetV2 (pre-trained) on {device}', flush=True)
    
    # start fine-tuning train process
    best_acc = 0.0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        
        # logging for why it fails
        writer.add_scalar('Loss/train', running_loss / len(trainloader), epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        
        # logging weight histograms
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                # focusing on depth-wise layers (groups=input_channels), these are the problematic cases
                if 'conv' in name and 'weight' in name:
                    writer.add_histogram(f'Weights/{name}', param, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Accuracy: {acc:.2f}% | Time: {time.time()-epoch_start:.1f}s")
        
        # save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{save_dir}/baseline/best_baseline.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
    
    print('Baseline training complete')
    writer.close()
    
    return best_acc
