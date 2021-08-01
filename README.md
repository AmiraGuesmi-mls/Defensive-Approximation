# Defensive-Approximation
Implementation of our proposed defense strategy against adversarial attacks "Defensive Approximation (DA)"

A Pytorch code for our [paper](https://dl.acm.org/doi/abs/10.1145/3445814.3446747). It includes the implementation of:

*   Our proposed approximate multiplier (Ax-FPM)
*   Approximate convolution layer
*   Approximate fully connected layer 
*   Tutorials on how to implement approximate CNN models

Note: for faster computation of convolution layers, we recommend using the Joblib Parallel tool (commented code in the approximate convolution layer) and Multiple Core CPU.

If you find this code useful in your research, please cite:

```
@inproceedings{10.1145/3445814.3446747,
author = {Guesmi, Amira and Alouani, Ihsen and Khasawneh, Khaled N. and Baklouti, Mouna and Frikha, Tarek and Abid, Mohamed and Abu-Ghazaleh, Nael},
title = {Defensive Approximation: Securing CNNs Using Approximate Computing},
year = {2021},
isbn = {9781450383172},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3445814.3446747},
doi = {10.1145/3445814.3446747},
booktitle = {Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems},
pages = {990–1003},
numpages = {14},
keywords = {Deep neural network, adversarial example, approximate computing, security},
location = {Virtual, USA},
series = {ASPLOS 2021}}
```
