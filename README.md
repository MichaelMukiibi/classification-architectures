# Image Classification Benchmark

An empirical framework for training, evaluating, and comparing diverse vision architectures (DenseNet, ResNet, ConvNeXt, Swin) across TensorFlow and PyTorch environments.

## Directory Layout

* `tensorflow_impl/`: Houses the TensorFlow code, dependency lists, and standalone dataset training executions.
* `pytorch_impl/`: Manages the PyTorch training routines and custom dataloading utilities.
* `architecture.md`: Contains the architectural deep-dives and engineering design choices for each network model.

## Prerequisites and Installation

### 1. Set Up the PyTorch Workspace
```bash
cd pytorch_impl
pip install -r requirements.txt
```

### 2. Set Up the TensorFlow Workspace
```bash
cd ../tensorflow_impl
pip install -r requirements.txt
```

## How to Run Execution Scripts

### Run PyTorch Benchmark
```bash
cd pytorch_impl
python train.py --model resnet --dataset cifar10 --epochs 10
```

### Run TensorFlow Benchmark
```bash
cd tensorflow_impl
python train.py --model resnet --dataset cifar10 --epochs 10
```

## Evaluation Metrics

The scripts output validation accuracy, tracking convergence loss curves, and logging per-class evaluation metrics upon training termination.