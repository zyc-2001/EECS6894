# Multi-Strategy NAS Optimization of EfficientNet

This project explores the performance of different EfficientNet models (B0 to B7) using Neural Architecture Search (NAS) for image classification tasks on the CIFAR-10 and Caltech-256 datasets. The objective is to analyze the scalability, accuracy, and efficiency of these models and propose strategies to optimize them for deployment on resource-constrained devices.

## Project Overview

Neural Architecture Search (NAS) has emerged as a key technique for automatically designing optimal deep neural network architectures. In this project, we aim to:

- Explore multi-objective optimization strategies for NAS, focusing on EfficientNet models.
- Evaluate the performance of EfficientNet models (B0 to B7) on CIFAR-10 and Caltech-256 datasets.
- Investigate hardware-aware NAS techniques to balance model complexity, accuracy, energy consumption, and latency.

## Features

- **EfficientNet B0-B7**: We evaluate the performance of EfficientNet models from B0 to B7 using NAS for optimizing both accuracy and efficiency.
- **Multi-Objective NAS**: We incorporate objectives such as accuracy, latency, and energy efficiency to create robust models suitable for real-world applications.
- **Hardware-Aware Optimization**: We optimize models to perform well on resource-constrained devices like edge devices and IoT devices.
- **Experiments on CIFAR-10 and Caltech-256**: Models are trained and evaluated on two benchmark datasets to compare their performance across different tasks.
