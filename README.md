# Graph pooling based on Graph U-Net

<img src="https://github.com/bknyaz/graph_nn/blob/master/figs/fig.png" height="192">

My attempt to reproduce graph classification results from recent papers [[1](https://openreview.net/forum?id=HJePRoAct7), [2](https://arxiv.org/abs/1811.01287)] using Graph U-Net. So far, my results using Graph U-Net are worse than the baseline (GCN).
I also compare to a recent work on Multigraph GCN (MGCN) [[4](https://arxiv.org/abs/1811.09595)].

This repository contains all necessary data for the PROTEINS dataset. It can be found [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets) along with similar datasets.

The baseline model is Graph Convolutional Network (GCN) [[3](https://arxiv.org/abs/1609.02907)].
The decoder part of Graph U-Net is not implemented yet in our code, i.e. the only difference with the baseline is using pooling based on dropping nodes between graph convolution layers.

Hyperparameters are taken from [[2](https://arxiv.org/abs/1811.01287)], but learning rate decay and dropout is also applied. The readout layer (last pooling layer over nodes) is also simplified to just ```max``` pooling over nodes.
All hyperparameters are the same for the baseline, Graph U-Net and Multigraph GCN (MGCN).

Implementation is very basic without much optimization, so that it is easier to debug and play around with the code.

```
python graph_unet.py --model gcn  # to run baseline GCN
python graph_unet.py --model unet  # to run Graph U-Net
python graph_unet.py --model mgcn  # to run Multigraph GCN
```

Repeating 10 times for different seeds:
```
for i in $(seq 1 10); do seed=$(( ( RANDOM % 10000 )  + 1 )); python graph_unet.py --model gcn --seed $seed | tee logs/gcn_proteins_"$i".log; done
```

Then reading log files can be done as following:
```
results_dir = './logs'
acc = []
for f in os.listdir(results_dir):
    with open(pjoin(results_dir, f), 'r') as fp:
        s = fp.readlines()[-1]        
    pos1 = s.find(':')
    acc.append(float(s[pos1+1:s[pos1:].find('(') + pos1]))
print(len(acc), np.mean(acc), np.std(acc))
```

Average and std of accuracy for 10-fold cross-validation. We also repeat experiments 10 times (as shown above) for different random seeds and report average and std over those 10 times.
| Model                 | PROTEINS | PROTEINS (10 times)
| --------------------- |:-------------:|:-------------:|
| GCN [[3](https://arxiv.org/abs/1609.02907)]                                   | 74.71 ± 3.44 | 74.37 ± 0.31 |
| GCN [[3](https://arxiv.org/abs/1609.02907)] + *A<sup>2</sup>*                 |  | |
| GCN [[3](https://arxiv.org/abs/1609.02907)] + *A<sup>2</sup>* + *2I*          |  | |
| Graph U-Net [[1](https://openreview.net/forum?id=HJePRoAct7), [2](https://arxiv.org/abs/1811.01287)]                           | 72.39 ± 3.34 |  |
| Graph U-Net [[1](https://openreview.net/forum?id=HJePRoAct7), [2](https://arxiv.org/abs/1811.01287)] + *A<sup>2</sup>*         |  | |
| Graph U-Net [[1](https://openreview.net/forum?id=HJePRoAct7), [2](https://arxiv.org/abs/1811.01287)] + *A<sup>2</sup>* + *2I*  |  | |
| Multigraph GCN (MGCN) [[4](https://arxiv.org/abs/1811.09595)]  | 74.62 ± 2.56 | 

# Requirements

The code is tested on Ubuntu 16.04 with pytorch 0.4.1 and Python 3.6, but should work in other environments with pytorch >= 0.4. 

The [python file](graph_unet.py) and [jupyter notebook file](graph_unet.ipynb) contain essentially the same code, the notebook file is kept for debugging purposes.

# References

[1] [Anonymous, Graph U-Net, submitted to ICLR 2019](https://openreview.net/forum?id=HJePRoAct7)

[2] [Cătălina Cangea, Petar Veličković, Nikola Jovanović, Thomas Kipf, Pietro Liò, Towards Sparse Hierarchical Graph Classifiers, NIPS Workshop on Relational Representation Learning, 2018](https://arxiv.org/abs/1811.01287)

[3] [Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017](https://arxiv.org/abs/1609.02907)

[4] [Boris Knyazev, Xiao Lin, Mohamed R. Amer, Graham W. Taylor, Spectral Multigraph Networks for Discovering and Fusing Relationships in Molecules, NIPS Workshop on Machine Learning for Molecules and Materials, 2018](https://arxiv.org/abs/1811.09595)
