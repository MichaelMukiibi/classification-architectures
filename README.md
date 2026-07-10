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

## Cloud Execution via Google Colab CLI

To run training benchmarks on a high-performance remote GPU without installing heavy machine learning frameworks or downloading datasets locally, use the Google Colab CLI toolchain.

### 1. Install the Colab CLI Tool
Install the execution wrapper globally or in an isolated tool environment using `uv`:
```bash
uv tool install google-colab-cli
```

### 2. Export W&B API key to current local terminal session
```bash
export $(cat .env | xargs)

```

### 3. Ephemeral One-Shot Execution (Recommended)
Trigger a remote, ephemeral NVIDIA T4 GPU instance, auto-forward local script execution arguments, and automatically tear down the compute node upon completion to prevent accidental credit over-use:
```bash
# Run PyTorch Benchmark Remotely
colab run --gpu T4 pytorch_impl/train.py --model resnet --dataset cifar10 --epochs 10 --wandb_key $WANDB_API_KEY

# Run TensorFlow Benchmark Remotely
colab run --gpu T4 tensorflow_impl/train.py --model resnet --dataset cifar10 --epochs 10 --wandb_key $WANDB_API_KEY
```
### 4. Persistent Session Execution (Alternative)

If you need to execute multiple experiments on the same active hardware instance without reloading the dataset repeatedly:

1. **Provision a dedicated runtime session**:
```bash
colab new -s benchmark-session --env WANDB_API_KEY=$WANDB_API_KEY --gpu T4
```
2. **Execute your local script directly on the active cluster node**:

```bash
colab exec -s benchmark-session --env WANDB_API_KEY=$WANDB_API_KEY -f pytorch_impl/train.py -- --model resnet --dataset cifar10 --epochs 10
```

3. **Terminate the session manually when finished**:
```bash
colab stop -s benchmark-session
```


## Evaluation Metrics

The scripts output validation accuracy, tracking convergence loss curves, and logging per-class evaluation metrics upon training termination.