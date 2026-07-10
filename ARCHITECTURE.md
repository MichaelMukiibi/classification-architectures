# Deep Learning Classification Architectures

This document details the design topologies, compute configurations, and comparative paradigms of classical and modern computer vision backbones.

---

## Architectural Breakdown

### 1. ResNet (Residual Networks)
* **Core Principle:** Employs explicit identity shortcut mappings to mitigate the vanishing gradient phenomenon in deep configurations.
* **Mechanics:** Modifies the layer mappings to track residual formulations instead of full underlying mappings.

### 2. DenseNet (Densely Connected Convolutional Networks)
* **Core Principle:** Maximizes information flow through dense, iterative layer-to-layer connectivity matrices.
* **Mechanics:** Concatenates feature maps from all prior layers along the channel axis, promoting continuous feature reuse and radical parameter reduction.

### 3. Swin Transformer (Shifted Window Transformer)
* **Core Principle:** Re-introduces localized spatial CNN inductive biases back into ViT backbones using hierarchical shifted multi-head self-attention windows.
* **Mechanics:** Restricts attention computations to non-overlapping local grids while enabling cross-window connectivity via shifted partitioning schemas.

### 4. ConvNeXt (Modernized ConvNet for the 2020s)
* **Core Principle:** Refactors traditional ResNet architectures step-by-step using structural lessons derived from Vision Transformers.
* **Mechanics:** Introduces a "patchify" stem layer, inverted bottlenecks, wide 7x7 depthwise convolutions, GELU activation functions, and singular LayerNorm placements.

---