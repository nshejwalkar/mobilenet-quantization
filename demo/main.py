"""
Layer-Selective INT8 Quantization for MobileNetV2
via Depthwise-Focused QAT

Authors: 
* Norman Zhang (nzhang11)
* Jay Katyan (jkatyan)
* Neel Shejwalkar (nshej)

Overview:
This script demonstrates layer-selective INT8 quantization
of the MobileNetV2 architecture using depth wise-focused
quantization-aware training (QAT).
"""


import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models.quantization import mobilenet_v2 as quantizable_mobilenet_v2
import argparse
import json
from utils import (
    evaluate,
    print_size_of_model,
    get_parameter_counts,
    plot_saturation,
    plot_layer_error,
    plot_channel_variance,
    compute_activation_drift,
    measure_throughput
)
from baseline import train_baseline_model
from ptq import run_ptq_per_tensor, run_ptq_per_channel
from qat import (
    run_full_qat,
    run_classifier_qat,
    run_depthwise_qat,
    run_depthwise_1x1_qat,
    run_depthwise_1x1_asymmetric_qat
)


#=========================================================
# Step 1: Initialize random seeds for reproducibility
#=========================================================
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)


#=========================================================
# Step 2: Load CIFAR-10 Dataset
#=========================================================

# Define transformations for training and testing sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load CIFAR-10 Training/Testing Datasets
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
calibset = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = DataLoader(trainset,
                         batch_size=256,
                         shuffle=True,
                         num_workers=4,
                         worker_init_fn=seed_worker,
                         generator=g)
testloader = DataLoader(testset,
                        batch_size=256,
                        shuffle=False,
                        num_workers=4,
                        worker_init_fn=seed_worker,
                        generator=g)
calibloader = DataLoader(trainset,
                         batch_size=256,
                         shuffle=True,
                         num_workers=4)


#=========================================================
# Step 3: Load Pre-trained MobileNetV2 Model
#=========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = quantizable_mobilenet_v2(weights='DEFAULT', quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model = model.to(device)


#=========================================================
# Main
#=========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Layer-Selective INT8 Quantization for MobileNetV2',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Training options
    parser.add_argument('--train-baseline', action='store_true',
                        help='Train the baseline FP32 model')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of epochs for baseline training (default: 60)')
    
    # Evaluation options
    parser.add_argument('--eval-baseline', action='store_true',
                        help='Evaluate the baseline FP32 model')
    
    # PTQ options
    parser.add_argument('--ptq-per-tensor', action='store_true',
                        help='Run Post-Training Quantization with per-tensor quantization')
    parser.add_argument('--ptq-per-channel', action='store_true',
                        help='Run Post-Training Quantization with per-channel quantization')
    
    # QAT options
    parser.add_argument('--qat-full', action='store_true',
                        help='Run full QAT (all layers)')
    parser.add_argument('--qat-classifier', action='store_true',
                        help='Run classifier-only QAT')
    parser.add_argument('--qat-depthwise', action='store_true',
                        help='Run depthwise-only QAT')
    parser.add_argument('--qat-dw-1x1', action='store_true',
                        help='Run depthwise + 1x1 projection QAT')
    parser.add_argument('--qat-dw-1x1-asym', action='store_true',
                        help='Run depthwise + 1x1 with asymmetric quantization QAT')
    parser.add_argument('--qat-epochs', type=int, default=10,
                        help='Number of epochs for QAT training (default: 10)')
    
    # Run all experiments
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    
    # General options
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to baseline checkpoint (default: results/baseline/best_baseline.pth)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup results directory
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f'{args.results_dir}/baseline', exist_ok=True)
    os.makedirs(f'{args.results_dir}/ptq', exist_ok=True)
    os.makedirs(f'{args.results_dir}/qat', exist_ok=True)
    
    # Determine checkpoint path
    if args.checkpoint is None:
        checkpoint_path = f'{args.results_dir}/baseline/best_baseline.pth'
    else:
        checkpoint_path = args.checkpoint
    
    # Track results
    results_data = {}
    
    # Train baseline if requested
    if args.train_baseline or (args.all and not os.path.exists(checkpoint_path)):
        print("\nTraining Baseline FP32 Model:")
        best_acc = train_baseline_model(model, trainloader, testloader, device, 
                                        num_epochs=args.epochs, save_dir=args.results_dir)
        print(f"\nBaseline training complete. Best accuracy: {best_acc:.2f}%")
        results_data['baseline_train'] = {'accuracy': best_acc}
    
    # Check if baseline model exists for other experiments
    if not os.path.exists(checkpoint_path):
        if not (args.train_baseline or args.all):
            print(f"\nError: Baseline model not found at {checkpoint_path}")
            print("Please train the baseline model first using --train-baseline")
            return
    
    # Evaluate baseline FP32 model
    if args.eval_baseline or args.all:
        print("\nEvaluating FP32 Baseline Model:")
        
        eval_model = quantizable_mobilenet_v2(weights=None, quantize=False)
        eval_model.classifier[1] = nn.Linear(eval_model.last_channel, 10)
        eval_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        eval_model = eval_model.to(device)
        
        acc_fp32 = evaluate(eval_model, testloader, device=device)
        size_fp32 = print_size_of_model(eval_model)
        print(f'Accuracy: {acc_fp32:.2f}%')
        print(f'Model size: {size_fp32:.2f} MB')
        
        results_data['fp32_baseline'] = {
            'accuracy': acc_fp32,
            'size_mb': size_fp32,
            'method': 'FP32 Baseline',
            'params_retrained': '--',
            'layers_updated': 'None',
            'quant_scheme': 'FP32'
        }
        
        # Save individual result
        with open(f'{args.results_dir}/baseline/eval_results.json', 'w') as f:
            json.dump(results_data['fp32_baseline'], f, indent=2)
    
    # PTQ Per-Tensor
    if args.ptq_per_tensor or args.all:
        print("\nRunning PTQ with Per-Tensor Quantization:")
        
        int8_model, acc_ptq, size_ptq = run_ptq_per_tensor(checkpoint_path, calibloader, testloader)
        print(f'\nResults:')
        print(f'  Accuracy: {acc_ptq:.2f}%')
        print(f'  Model size: {size_ptq:.2f} MB')
        
        results_data['ptq_per_tensor'] = {
            'accuracy': acc_ptq,
            'size_mb': size_ptq,
            'method': 'PTQ (Per-Tensor)',
            'params_retrained': '0%',
            'layers_updated': 'None',
            'quant_scheme': 'Symmetric'
        }
        
        # Save model and results
        torch.save(int8_model.state_dict(), f'{args.results_dir}/ptq/ptq_per_tensor.pth')
        with open(f'{args.results_dir}/ptq/ptq_per_tensor_results.json', 'w') as f:
            json.dump(results_data['ptq_per_tensor'], f, indent=2)
    
    # PTQ Per-Channel
    if args.ptq_per_channel or args.all:
        print("\nRunning PTQ with Per-Channel Quantization:")
        
        int8_model, acc_ptq, size_ptq = run_ptq_per_channel(checkpoint_path, calibloader, testloader)
        print(f'\nResults:')
        print(f'  Accuracy: {acc_ptq:.2f}%')
        print(f'  Model size: {size_ptq:.2f} MB')
        
        results_data['ptq_per_channel'] = {
            'accuracy': acc_ptq,
            'size_mb': size_ptq,
            'method': 'PTQ (Per-Channel)',
            'params_retrained': '0%',
            'layers_updated': 'None',
            'quant_scheme': 'Symmetric'
        }
        
        # Save model and results
        torch.save(int8_model.state_dict(), f'{args.results_dir}/ptq/ptq_per_channel.pth')
        with open(f'{args.results_dir}/ptq/ptq_per_channel_results.json', 'w') as f:
            json.dump(results_data['ptq_per_channel'], f, indent=2)
    
    # Load baseline for QAT experiments if any QAT experiment is requested
    qat_experiments = [args.qat_full, args.qat_classifier, args.qat_depthwise, 
                       args.qat_dw_1x1, args.qat_dw_1x1_asym]
    
    if any(qat_experiments) or args.all:
        baseline_model = quantizable_mobilenet_v2(weights=None, quantize=False)
        baseline_model.classifier[1] = nn.Linear(baseline_model.last_channel, 10)
        baseline_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # QAT Full
    if args.qat_full or args.all:
        print("\nRunning Full QAT:")
        
        int8_model, acc_qat, size_qat, trainable, total = run_full_qat(baseline_model, trainloader, testloader, 
                                                      device, epochs=args.qat_epochs)
        
        print(f'\nResults:')
        print(f'  Accuracy: {acc_qat:.2f}%')
        print(f'  Model size: {size_qat:.2f} MB')
        print(f'  Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})')
        
        results_data['qat_full'] = {
            'accuracy': acc_qat,
            'size_mb': size_qat,
            'method': 'Full QAT',
            'params_retrained': f'{trainable/total:.1%}',
            'layers_updated': 'All Layers',
            'quant_scheme': 'Symmetric'
        }
        
        # Save model and results
        torch.save(int8_model.state_dict(), f'{args.results_dir}/qat/qat_full.pth')
        with open(f'{args.results_dir}/qat/qat_full_results.json', 'w') as f:
            json.dump(results_data['qat_full'], f, indent=2)
    
    # QAT Classifier
    if args.qat_classifier or args.all:
        print("\nRunning Classifier-Only QAT:")
        
        int8_model, acc_qat, size_qat, trainable, total = run_classifier_qat(baseline_model, trainloader, testloader, 
                                                            device, epochs=args.qat_epochs)
        
        print(f'\nResults:')
        print(f'  Accuracy: {acc_qat:.2f}%')
        print(f'  Model size: {size_qat:.2f} MB')
        print(f'  Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})')
        
        results_data['qat_classifier'] = {
            'accuracy': acc_qat,
            'size_mb': size_qat,
            'method': 'QAT (Classifier)',
            'params_retrained': f'{trainable/total:.1%}',
            'layers_updated': 'FC Only',
            'quant_scheme': 'Symmetric'
        }
        
        # Save model and results
        torch.save(int8_model.state_dict(), f'{args.results_dir}/qat/qat_classifier.pth')
        with open(f'{args.results_dir}/qat/qat_classifier_results.json', 'w') as f:
            json.dump(results_data['qat_classifier'], f, indent=2)
    
    # QAT Depthwise
    if args.qat_depthwise or args.all:
        print("\nRunning Depthwise-Only QAT:")
        
        int8_model, acc_qat, size_qat, trainable, total = run_depthwise_qat(baseline_model, trainloader, testloader, 
                                                           device, epochs=args.qat_epochs)
        
        print(f'\nResults:')
        print(f'  Accuracy: {acc_qat:.2f}%')
        print(f'  Model size: {size_qat:.2f} MB')
        print(f'  Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})')
        
        results_data['qat_depthwise'] = {
            'accuracy': acc_qat,
            'size_mb': size_qat,
            'method': 'QAT (Depthwise)',
            'params_retrained': f'{trainable/total:.1%}',
            'layers_updated': 'DW',
            'quant_scheme': 'Symmetric'
        }
        
        # Save model and results
        torch.save(int8_model.state_dict(), f'{args.results_dir}/qat/qat_depthwise.pth')
        with open(f'{args.results_dir}/qat/qat_depthwise_results.json', 'w') as f:
            json.dump(results_data['qat_depthwise'], f, indent=2)
    
    # QAT Depthwise + 1x1
    if args.qat_dw_1x1 or args.all:
        print("\nRunning Depthwise + 1x1 QAT:")
        
        int8_model, acc_qat, size_qat, trainable, total = run_depthwise_1x1_qat(baseline_model, trainloader, testloader, 
                                                               device, epochs=args.qat_epochs)
        
        print(f'\nResults:')
        print(f'  Accuracy: {acc_qat:.2f}%')
        print(f'  Model size: {size_qat:.2f} MB')
        print(f'  Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})')
        
        results_data['qat_dw_1x1'] = {
            'accuracy': acc_qat,
            'size_mb': size_qat,
            'method': 'QAT (DW + 1x1)',
            'params_retrained': f'{trainable/total:.1%}',
            'layers_updated': 'DW, 1x1 PW',
            'quant_scheme': 'Symmetric'
        }
        
        # Save model and results
        torch.save(int8_model.state_dict(), f'{args.results_dir}/qat/qat_dw_1x1.pth')
        with open(f'{args.results_dir}/qat/qat_dw_1x1_results.json', 'w') as f:
            json.dump(results_data['qat_dw_1x1'], f, indent=2)
    
    # QAT Depthwise + 1x1 Asymmetric
    if args.qat_dw_1x1_asym or args.all:
        print("\nRunning Depthwise + 1x1 Asymmetric QAT:")
        
        int8_model, acc_qat, size_qat, trainable, total = run_depthwise_1x1_asymmetric_qat(baseline_model, trainloader, 
                                                                          testloader, device, 
                                                                          epochs=args.qat_epochs)
        
        print(f'\nResults:')
        print(f'  Accuracy: {acc_qat:.2f}%')
        print(f'  Model size: {size_qat:.2f} MB')
        print(f'  Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})')
        
        results_data['qat_dw_1x1_asym'] = {
            'accuracy': acc_qat,
            'size_mb': size_qat,
            'method': 'QAT (DW + 1x1 + Asym)',
            'params_retrained': f'{trainable/total:.1%}',
            'layers_updated': 'DW, 1x1 PW',
            'quant_scheme': 'Asymmetric'
        }
        
        # Save model and results
        torch.save(int8_model.state_dict(), f'{args.results_dir}/qat/qat_dw_1x1_asym.pth')
        with open(f'{args.results_dir}/qat/qat_dw_1x1_asym_results.json', 'w') as f:
            json.dump(results_data['qat_dw_1x1_asym'], f, indent=2)
    
    # Generate summary if any experiments were run
    if results_data:
        print("\nResults Summary:")
        
        # Create summary table
        summary_rows = []
        for key, data in results_data.items():
            if key != 'baseline_train':
                row = [
                    data['method'],
                    data['params_retrained'],
                    data['layers_updated'],
                    data['quant_scheme'],
                    f"{data['size_mb']:.2f} MB",
                    f"{data['accuracy']:.2f}%"
                ]
                summary_rows.append(row)
        
        if summary_rows:
            df = pd.DataFrame(summary_rows, columns=[
                "Method", "Params Retrained", "Layers Updated", 
                "Quant Scheme", "Model Size", "Accuracy"
            ])
            
            print(df.to_string(index=False))
            
            # Save summary
            df.to_csv(f'{args.results_dir}/summary.csv', index=False)
            print(f"\nSummary saved to '{args.results_dir}/summary.csv'")
            
            # Save full results as JSON
            with open(f'{args.results_dir}/full_results.json', 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Full results saved to '{args.results_dir}/full_results.json'")


if __name__ == "__main__":
    main()