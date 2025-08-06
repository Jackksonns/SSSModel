# SSSModel: A Lightweight CIFAR-10 Image Classification Network Based on SCSA and RFAConv Fusion

## Project Introduction

SSSModel is a lightweight convolutional neural network designed for CIFAR-10 classification. It employs the RFAConv module for initial feature extraction, overcoming the limitations of large convolutional kernels by focusing on spatial information within the receptive field. The SCSA module enhances these features using shared multi-semantic spatial attention (SMSA) and progressive channel self-attention (PCSA), leveraging spatial priors to guide channel attention and reduce multi-semantic discrepancies. The network concludes with global average pooling, flattening, and a linear classification layer to output 10-class predictions.

## Model Architecture

- **RFAConv Module**: Applies a novel receptive field attention mechanism to weight spatial features, enhancing local feature extraction with low computational cost and providing robust initial representations.
- **SCSA Module**: Integrates SMSA and PCSA sequentially. SMSA captures multi-scale spatial information using depthwise separable 1D convolutions and injects priors via group normalization. PCSA recalibrates features in the channel dimension using SMSA-derived spatial priors, optimizing spatial-channel synergy for comprehensive feature capture.
- **Output Layer**: Aggregates spatial features across channels with global average pooling, reduces parameters, and increases global sensitivity, followed by flattening and a linear layer mapping to CIFAR-10’s 10 classes.

## Training Strategies

- **K-fold Cross-Validation**: Rotates validation sets to evaluate generalization and mitigate overfitting.
- **MixUp Data Augmentation**: Mixes images and labels to generate new samples, enhancing generalization.
- **Cosine Annealing Learning Rate Scheduling**: Starts with a high learning rate for rapid convergence, gradually decreasing to escape local minima.
- **Loss Function Combination**: Combines multiple losses to optimize accuracy and feature distribution.
- **Test-Time Augmentation (TTA)**: Averages predictions from augmented test samples to reduce variance.
- **Model Ensembling**: Integrates multiple models via majority voting for improved performance.

> **Note**: JKmodel achieves ~85% accuracy on CIFAR-10, below ResNet-18, but validates attention fusion feasibility and offers design insights.

## Running the Code

1. Install PyTorch and torchvision (CUDA recommended).

2. Clone and navigate:

   ```bash
   git clone https://github.com/Jackksonns/SSSModel
   cd SSSModel
   ```

3. Train:

   ```bash
   python train.py
   ```

4. Ensemble inference:

   ```bash
   python training_process_optimization.py
   ```

## Experiments and Conclusions

- **Setup**: Trained on CIFAR-10 with K-fold cross-validation over 50 epochs, yielding ~85% single-model accuracy and ~86% with ensembling.
- **Conclusions**:
  - RFAConv and SCSA enhance feature capture in lightweight networks.
  - Performance lags behind classic architectures on small-scale tasks, but shows potential.
  - Experiment focuses on design and training practice; results are referential, not production-ready.

## Future Work

Future research will adapt attention fusion for tasks like image restoration (e.g., denoising, super-resolution) by modifying architecture and loss functions to assess generalizability across domains.

## References & Projects

1. Si, Shunming, et al. *SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention*. arXiv:2407.05128, 2024. [https://arxiv.org/abs/2407.05128](https://arxiv.org/abs/2407.05128)
2. Zhang, Daquan, et al. *RFAConv: Innovating Spatial Attention and Standard Convolutional Operation*. arXiv:2304.03198, 2023. [https://arxiv.org/abs/2304.03198](https://arxiv.org/abs/2304.03198)
3. Zhang, Hongyi, et al. *mixup: Beyond Empirical Risk Minimization*. arXiv:1710.09412, 2017. [https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412)
4. Loshchilov, Ilya, et al. *SGDR: Stochastic Gradient Descent with Warm Restarts*. arXiv:1608.03983, 2016. [https://arxiv.org/abs/1608.03983](https://arxiv.org/abs/1608.03983)
5. SHENZHENYI. *Classify-Leaves-Kaggle-MuLi-d2l-course*. GitHub, 2021. [https://github.com/SHENZHENYI/Classify-Leaves-Kaggle-MuLi-d2l-course](https://github.com/SHENZHENYI/Classify-Leaves-Kaggle-MuLi-d2l-course)
