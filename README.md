# EfficientNet (B0-B7) with CIFAR-10 Classification

This project explores the use of EfficientNet models (from B0 to B7) for image classification on the CIFAR-10 dataset. The models are trained with mixed precision to improve performance and reduce memory usage.

## Features

- **EfficientNet B0 to B7**: Pre-trained models from the EfficientNet family (B0 to B7) on ImageNet used for CIFAR-10 classification.
- **Mixed Precision Training**: Increases efficiency by using `float16` for certain operations, saving memory and speeding up training.
- **CIFAR-10 Dataset**: A well-known dataset consisting of 32x32 color images across 10 different classes.
- **Custom Top Layers**: Adaptation of EfficientNet models to work with the CIFAR-10 dataset.
- **Model Checkpoints**: Automatically saves the best performing model during training for future use.

## Usage

To experiment with different EfficientNet models (B0 to B7), you can easily switch the model in the script by changing the line where the base model is loaded.

