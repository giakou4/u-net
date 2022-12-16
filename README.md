<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/u-net/main/assets/bg.jpg">
</p>

# U-NET

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/u-net/LICENSE)

## Network Architecture

The network architecture is illustrated in Figure. It consists of a **contracting**
path (left side) and an **expansive path** (right side). 

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/u-net/main/assets/unet.png">
</p>

The **contracting** path follows
the typical architecture of a convolutional network. It consists of:
* Convolutions (kernel size = ```3x3```, stride = ```1``` padding = ```1```)
* Batch Normalization (added in here, does not exist in original implementation)
* ReLU
* Convolutions (kernel size = ```3x3```, stride = ```1``` padding = ```1```)
* Batch Normalization (added in here, does not exist in original implementation)
* ReLU
* Downsampling: Max pooling operation (kernel size = ```2x2```, stride = ```2```)


Every step in the **expansive path** consists of:
* Upsampling: Transpose Convolution (kernel size = ```2x2```, stride = ```2```, padding = ```0```)
* Convolutions (filter size = ```2x2```)
* Concatenation with the correspondingly cropped feature map from the contracting path
* Convolutions (kernel size = ```3x3```)
* ReLU
* Convolutions (kernel size = ```3x3```)
* ReLU


At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. total the network has 23 convolutional layers.

## Training

* Stochastic Gradient Descent (SGD)
* Batch Size = ```1```
* Momentum = ```0.99```
* Loss: Cross-Entropy Loss
* Data Augmentation: shift, rotation, ranodm elastic deformation (random displacement, per-pixel displacements)
* The output image is smaller than the input by a constant border width
