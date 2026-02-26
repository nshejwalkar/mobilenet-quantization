# MobileNetV2 Quantization Demo

This project demonstrates layer-selective INT8 quantization of MobileNetV2 using various quantization techniques.

## Project Structure

```
demo/
├── main.py          # Main orchestration script with CLI
├── baseline.py      # Baseline FP32 training module
├── ptq.py           # Post-Training Quantization experiments
├── qat.py           # Quantization-Aware Training experiments
├── utils.py         # Utility functions (evaluation, plotting, metrics)
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Setup

### Installation

```bash
pip install -r requirements.txt
```

### Dataset

The script automatically downloads the CIFAR-10 dataset on first run. The dataset will be saved to `./data/` directory (approximately 170 MB).

### Quick Start

Run a quick test to verify everything works:
```bash
# Train baseline for 1 epoch (quick test)
python main.py --train-baseline --epochs 1

# Evaluate the baseline
python main.py --eval-baseline

# Run a quick PTQ experiment (no training required)
python main.py --ptq-per-channel
```

## Usage

The script supports command-line flags to run specific experiments or all experiments at once.

### Basic Commands

```bash
# Show help
python main.py --help

# Train baseline FP32 model
python main.py --train-baseline

# Evaluate baseline FP32 model
python main.py --eval-baseline

# Run PTQ experiments
python main.py --ptq-per-tensor
python main.py --ptq-per-channel

# Run QAT experiments
python main.py --qat-full
python main.py --qat-classifier
python main.py --qat-depthwise
python main.py --qat-dw-1x1
python main.py --qat-dw-1x1-asym

# Run all experiments
python main.py --all
```

### Command-Line Flags

#### Training Options
- `--train-baseline` - Train the baseline FP32 model
- `--epochs N` - Number of epochs for baseline training (default: 60)

#### Evaluation Options
- `--eval-baseline` - Evaluate the baseline FP32 model

#### PTQ Options
- `--ptq-per-tensor` - Run Post-Training Quantization with per-tensor quantization
- `--ptq-per-channel` - Run Post-Training Quantization with per-channel quantization

#### QAT Options
- `--qat-full` - Run full QAT (all layers)
- `--qat-classifier` - Run classifier-only QAT
- `--qat-depthwise` - Run depthwise-only QAT
- `--qat-dw-1x1` - Run depthwise + 1x1 projection QAT
- `--qat-dw-1x1-asym` - Run depthwise + 1x1 with asymmetric quantization QAT
- `--qat-epochs N` - Number of epochs for QAT training (default: 10)

#### General Options
- `--all` - Run all experiments
- `--results-dir PATH` - Directory to save results (default: ./results)
- `--checkpoint PATH` - Path to baseline checkpoint (default: results/baseline/best_baseline.pth)

### Examples

```bash
# Train baseline and evaluate
python main.py --train-baseline --eval-baseline

# Run specific experiments
python main.py --eval-baseline --ptq-per-channel --qat-depthwise

# Run all experiments with custom results directory
python main.py --all --results-dir ./my_results

# Use existing checkpoint for experiments
python main.py --checkpoint ./checkpoints/baseline.pth --qat-full --qat-depthwise
```

## Results Organization

All results are saved in the `results/` directory (or custom directory specified by `--results-dir`):

```
results/
├── baseline/
│   ├── best_baseline.pth       # Trained baseline model
│   ├── eval_results.json       # Baseline evaluation results
│   └── runs/                   # TensorBoard logs
├── ptq/
│   ├── ptq_per_tensor.pth      # PTQ per-tensor model
│   ├── ptq_per_tensor_results.json
│   ├── ptq_per_channel.pth     # PTQ per-channel model
│   └── ptq_per_channel_results.json
├── qat/
│   ├── qat_full.pth            # Full QAT model
│   ├── qat_full_results.json
│   ├── qat_classifier.pth      # Classifier-only QAT model
│   ├── qat_classifier_results.json
│   ├── qat_depthwise.pth       # Depthwise QAT model
│   ├── qat_depthwise_results.json
│   ├── qat_dw_1x1.pth          # Depthwise + 1x1 QAT model
│   ├── qat_dw_1x1_results.json
│   ├── qat_dw_1x1_asym.pth     # Asymmetric QAT model
│   └── qat_dw_1x1_asym_results.json
├── summary.csv                  # Summary of all experiments
└── full_results.json           # Complete results in JSON format
```

### Result Files

- **Model files (`.pth`)**: PyTorch state dictionaries for each trained/quantized model
- **Individual results (`.json`)**: Detailed metrics for each experiment including accuracy, model size, quantization scheme, etc.
- **`summary.csv`**: Tabular summary of all experiments for easy comparison
- **`full_results.json`**: Complete results in JSON format for programmatic access

## Experiment Descriptions

### Baseline
- **FP32 Baseline**: Pre-trained MobileNetV2 fine-tuned on CIFAR-10 with full 32-bit precision

### PTQ (Post-Training Quantization)
- **Per-Tensor**: Quantizes using a single scale/zero-point per tensor
- **Per-Channel**: Quantizes using separate scale/zero-point per output channel

### QAT (Quantization-Aware Training)
- **Full QAT**: Fine-tunes all layers with quantization simulation (100% of parameters)
- **Classifier QAT**: Only fine-tunes the classifier layer (0.6% of parameters)
- **Depthwise QAT**: Fine-tunes depthwise convolution layers (3.4% of parameters)
- **DW + 1x1 QAT**: Fine-tunes depthwise and 1x1 projection layers (46.5% of parameters)
- **DW + 1x1 Asymmetric**: Same as above but with asymmetric quantization for activations

## Requirements

```bash
pip install torch torchvision pandas matplotlib tensorboard
```

## Code Organization

The codebase is organized into modular components:

- **`main.py`**: Entry point with CLI argument parsing and experiment orchestration. Handles all result printing and file I/O.
- **`baseline.py`**: Implements baseline FP32 model training with TensorBoard logging.
- **`ptq.py`**: Implements Post-Training Quantization with per-tensor and per-channel schemes.
- **`qat.py`**: Implements five QAT ablation studies with different layer-selective strategies.
- **`utils.py`**: Shared utility functions for model evaluation, size calculation, parameter counting, and visualization.

Each module focuses on a specific aspect of the quantization pipeline, making the code easier to maintain and extend.

## Notes

- The script automatically downloads CIFAR-10 dataset on first run
- GPU training is automatically used if available
- All experiment functions are modular and can be imported independently
- Parameter retraining percentages are computed dynamically based on actual model structure
