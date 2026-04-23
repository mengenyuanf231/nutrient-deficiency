# DDCLB-LDNet: A Lightweight Model for Strawberry Nutrient Deficiency Diagnosis via Bidirectional Manifold Interaction
This repository provides the official implementation of the paper:

**DDCLB-LDNet: A Lightweight Model for Strawberry Nutrient Deficiency Diagnosis via Bidirectional Manifold Interaction**

---

## Overview
DDCLB-LDNet is designed for accurate nutrient deficiency diagnosis of dense strawberry leaves under complex greenhouse environments. Aiming at dense occlusion, small symptom targets, irregular boundaries and weak phenotypic features, this study selects YOLOv10s as the baseline model, and constructs a lightweight end-to-end detection network with bidirectional manifold feature interaction.

The proposed model consists of three core components:
- **DDGNet**: Dynamic Dual-Domain Hierarchical Graph Network (feature extraction backbone)
- **BDFMN**: Bidirectional Dynamic Fusion Mixing Neck (bidirectional manifold interaction neck)
- **DL-DEHead**: Dynamic Label-Optimized Lightweight Detail Enhancement Detection Head

DDCLB-LDNet effectively improves detection performance for dense, small, low-contrast and morphologically diverse nutrient deficiency symptoms, including **Ca, Fe, P deficiency** of strawberry leaves, while maintaining lightweight parameters and low computational cost.

The model adopts front-end multi-scale edge feature enhancement, dual-domain spatial-frequency filtering, bidirectional cross-layer manifold feature interaction, dynamic heterogeneous convolution optimization and adaptive dynamic label assignment strategy. It realizes collaborative optimization of feature extraction, cross-layer fusion and target localization, and provides an efficient lightweight solution for strawberry nutrient deficiency detection under dense occlusion conditions.

---

## Repository Structure
```text
.
├── block.py              # Implementation of DDGNet, BDFMN and all network modules
├── loss.py               # Loss function definitions & dynamic label assignment strategy
├── all.yaml              # DDCLB-LDNet model architecture and configuration file
├── train.py              # Training script
├── test.py               # Testing / inference script
└── README.md


---

## Requirements

- Python 3.8+
- PyTorch
- CUDA (optional, recommended for training and inference)

Please install the required packages according to your environment.

---

## Dataset

The dataset used in this study is publicly available at:

## Dataset

The dataset used in this study is publicly available [here](https://drive.google.com/drive/folders/1HeB-U_AUlxknExO_jJNN7_J5-YbYxvSS?usp=sharing).

After downloading the dataset, please unzip it and place the `datasets/` directory in the root of this repository.



---

## Training

To train DSLNDD-Net on your own machine:

1. Download the dataset from the link above.
2. Unzip the dataset and move the `datasets/` folder to the root directory of this repository.
3. Check the configuration file and training settings.
4. Run:

~~~bash
python train.py
~~~

---

## Testing

To evaluate the model or perform inference, please check the arguments in `test.py` and run:

~~~bash
python test.py
~~~

---

## Code Availability

This code is released for academic research purposes only.
