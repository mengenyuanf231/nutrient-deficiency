DDCLB-LDNet: A Lightweight Model for Strawberry Nutrient Deficiency Diagnosis via Bidirectional Manifold Interaction
This repository provides the official implementation of the paper:
DDCLB-LDNet: A Lightweight Model for Strawberry Nutrient Deficiency Diagnosis via Bidirectional Manifold Interaction
Overview
DDCLB-LDNet is proposed for accurate nutrient deficiency diagnosis of dense strawberry leaves in complex greenhouse environments. Aiming at the problems of dense occlusion, small symptom targets, irregular boundaries and weak phenotypic features, this study takes YOLOv10s as the baseline model, and constructs a lightweight end-to-end detection network with bidirectional manifold feature interaction.
The framework integrates three core components:
DDGNet: Dynamic Dual-Domain Hierarchical Graph Network (feature extraction backbone)
BDFMN: Bidirectional Dynamic Fusion Mixing Neck (bidirectional manifold interaction neck)
DL-DEHead: Dynamic Label-Optimized Lightweight Detail Enhancement Detection Head
DDCLB-LDNet effectively improves the detection performance of dense, small, low-contrast and morphologically diverse nutrient deficiency symptoms, including Ca, Fe, P deficiency of strawberry leaves, while maintaining lightweight parameters and low computational cost.
The model adopts front-end multi-scale edge feature enhancement, dual-domain spatial-frequency filtering, bidirectional cross-layer manifold feature interaction, dynamic heterogeneous convolution optimization and adaptive dynamic label assignment strategy. It achieves collaborative optimization of feature extraction, cross-layer fusion and target localization, and provides an efficient lightweight solution for strawberry nutrient deficiency detection under dense occlusion conditions.
