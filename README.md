
# Attention-based Channel-wise Pruning with Quantization-aware Training

This repository contains the implementation of attention-based channel-wise pruning with quantization-aware training (QAT) for neural networks, for the final project of the ELE6310E course.

## Introduction

Deep Neural Networks (DNNs) often face challenges in deployment on resource-constrained edge devices due to their large size and computational demands. To address this issue, techniques such as quantization (reducing precision) and pruning (removing unimportant network components) are used to optimize network size, speed, energy consumption, and storage without significant loss of accuracy.

This project explores attention-based channel-wise pruning combined with QAT to reduce model size while maintaining high accuracy.

## Key Features

- **Channel-wise Pruning**: Unlike traditional layer-wise pruning, this method prunes individual channels in a structured manner, which is more hardware-friendly.
  
- **Quantization-aware Training (QAT)**: Implements quantization-aware techniques, ensuring that the network is trained to perform well under reduced precision.

- **Loss Function Modification**: Includes modifications to the loss function to incorporate attention values and encourage maximum channel-wise pruning without crucial accuracy degradation.

- **Comprehensive Training Pipeline**: The project implements a complete training pipeline that integrates attention-based pruning, quantization-aware training, and recovery mechanisms to optimize model size and accuracy.

## Repository Structure

<pre>
.
├── ELE6310E_final_report.pdf  # Full Description of the project
├── main.ipynb                 # Main notebook for running experiments, training models and results analysis.
├── prune_Q
│   ├── load_data.py           # Code for loading the CIFAR-10 dataset.
│   ├── quant_utils.py         # Code for implementing quantization and dequantization.
│   ├── custom_modules.py      # Code for implementing custom quantized attention-based Conv2d and Linear modules.
│   └── resnet.py              # Code for building the ResNet32 network using custom modules.
└── Saved
    ├── Training_logs          # Folder for storing training logs.
    └── Final_model            # Folder for storing the final saved model.
</pre>



## Usage

To run the training and evaluation scripts:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Muhammed-Mamdouh/Channel-Wise-Attention-based-pruning-with-Quantization-aware-training.git
   ```

2. Set up the environment with the required dependencies specified in the requirements file (Python, PyTorch, etc.).

3. Navigate to the `main.ipynb` directory and specify the training parameters in the [`training`](https://nbviewer.org/github/Muhammed-Mamdouh/Channel-Wise-Attention-based-pruning-with-Quantization-aware-training/blob/master/main.ipynb#Training) section.

4. View the experiment results in the Jupyter notebooks provided in the [`results`](https://nbviewer.org/github/Muhammed-Mamdouh/Channel-Wise-Attention-based-pruning-with-Quantization-aware-training/blob/master/main.ipynb#Results-Analysis) section.

## Results and Analysis

The repository includes detailed experimental results, visualizations, and analysis in the Jupyter notebooks. Check the notebooks for insights into model compression, accuracy, and performance trade-offs.

## Contributors

- Muhammed Mamdouh Salah Abdelefatah
- Matilda Novshadian


## References
- Qiang He, Wenrui Shi, and Ming Dong. “Learning Pruned Structure and Weights Simul- taneously from Scratch: an Attention based Approach”. In: (2021). arXiv: 2111 . 02399 [cs.LG]. URL: http://arxiv.org/abs/2111.02399.
- Hui Zou and Trevor Hastie. “Regularization and Variable Selection via the Elastic Net”. In: Journal of the Royal Statistical Society. Series B (Statistical Methodology) 67.2 (2005), pp. 301–320.
-Yoshua Bengio, Nicholas L  ́eonard, and Aaron Courville. “Estimating or Propagating Gradi- ents through Stochastic Neurons for Conditional Computation”. In: arXiv preprint arXiv:1308.3432 (2013).

For any questions or inquiries, please feel free to contact the contributors.
