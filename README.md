# AlignDiff
This is a PyTorch implementation for our <AlignDiff: Gradient-Semantic Aligned Diffusion Model for Sequential Recommendation> paper.
## oveview
Sequential recommendation predicts users' next interactions by modeling dynamically evolving preferences. Recently, diffusion-based sequence recommenders have improved recommendation accuracy through distribution modeling, but they face the dual problems of distribution alignment and semantic guidance distortion. We propose AlignDiff, a Gradient-Semantic Aligned Diffusion Model for Sequential Recommendation addressing these issues via two innovations: (1) By combining a denoising predictor and an energy function network into a siamese denoising network, this network learns the gradient differences between distributions using cross-entropy loss and gradient score-matching loss, explicitly constraining the predicted denoising distribution to fit the ground-truth data distribution; (2) Multi-Level conditional guidance fusing sequence embeddings with attention-derived deep semantic features, efficiently modeling user preferences and correcting the problem of distortion in guidance conditions by mining deep semantic information in user interaction sequences, which guides the model to denoise in the direction of the correct denoising trajectory. Experiments demonstrate that AlignDiff significantly outperforms all baselines on three datasets.

## Requirements
Python 3.8
PyTorch 1.11.0
numpy 1.23.4
Our code has been tested running under a Linux desktop with NVIDIA GeForce RTX 3090 GPU.

## Training
python main.py --dataset amazon_beauty
