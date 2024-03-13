# Convolutional auto-encoder with variational information bottleneck (CAE-VIB)

This repository contains code to replicate the results of the paper [Data-Efficient Dimensionality Reduction and Surrogate Modeling of High-Dimensional Stress Fields](). The notebook discusses the application of [DeepHyper](https://ieeexplore.ieee.org/abstract/document/8638041) augmented convolutional auto-encoder enhanced by variational information bottleneck model (CAE-VIB) for modeling high-dimensional stress-field data. The dataset is generated using ANSYS APDL. Further details about the dataset and the modeling can be found in the paper.

# Menu

The following are the experiments performed in the paper:

- [Results_1](https://github.com/AnirbanSamaddar/CAE-VIB/tree/main/Results_1/VIB_hpo_ae2d_eval-single-seed.ipynb): This demonstrates the training of the DeepHyper augmented CAE-VIB method and generates the plots of Sec. 4.1 for the non-rotated dataset.
- [Results_2](https://github.com/AnirbanSamaddar/CAE-VIB/blob/main/Results_2/VIB_hpo_ae2d_pixel-importance.ipynb): This demonstrates calculating the pixel importance presented in Sec. 4.3 and generates the plots.

# Package requirements:

Below is a list of main packages and their versions required to run the notebook. Please note that the list may not be exhaustive.

```
- tensorflow: 2.10.0
- keras: 2.10.0
- deephyper: 0.6.0
- numpy: 1.22.0
- matplotlib: 3.4.2
```
