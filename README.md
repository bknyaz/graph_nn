# Graph pooling based on Graph U-Net

<img src="https://github.com/bknyaz/graph_nn/blob/master/figs/fig.png" height="192">

My attempt to reproduce graph classification results from recent papers [[1](https://openreview.net/forum?id=HJePRoAct7), [2](https://arxiv.org/abs/1811.01287)] using Graph U-Net. So far, my results are worse than the baseline.

This repository contains all necessary data for the PROTEINS dataset. It can be found [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) along with similar datasets.

The baseline model is Graph Convolutional Network (GCN) [[3](https://arxiv.org/abs/1609.02907)].
The decoder part of Graph U-Net is not implemented yet in our code, i.e. the only difference with the baseline is using pooling based on dropping nodes between graph convolution layers.

Hyperparameters are taken from [[2](https://arxiv.org/abs/1811.01287)], but learning rate decay and dropout is also applied. The readout layer (last pooling layer over nodes) is also simplified to just ```max``` pooling over nodes.
All hyperparameters are the same for the baseline and Graph U-Net.

```
python graph_unet.py --model gcn  # to run baseline GCN
python graph_unet.py --model unet  # to run Graph U-Net
```

| Model                 | PROTEINS          
| --------------------- |:-------------:|
| GCN                                   | 76.09 ± 0.69 |
| GCN + *A<sup>2</sup>*                 | 75.76 ± 0.54 |
| GCN + *A<sup>2</sup>* + *2I*          | 75.35 ± 0.57 |
| Graph U-Net                           | 72.95 ± 1.09 |
| Graph U-Net + *A<sup>2</sup>*         | 74.18 ± 0.92 |
| Graph U-Net + *A<sup>2</sup>* + *2I*  | 73.56 ± 0.64 |


# References

[1] [Anonymous, Graph U-Net, submitted to ICLR 2019](https://openreview.net/forum?id=HJePRoAct7)

[2] [Cătălina Cangea, Petar Veličković, Nikola Jovanović, Thomas Kipf, Pietro Liò, Towards Sparse Hierarchical Graph Classifiers, NIPS Workshop on Relational Representation Learning, 2018](https://arxiv.org/abs/1811.01287)

[3] [Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017](https://arxiv.org/abs/1609.02907)
