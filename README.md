# PFGE
This repository contains a PyTorch implementation of the Parsimonious Fast Geometric Ensembling (PFGE) procedures from the paper

[PFGE: Parsimonious Fast Geometric Ensembling of DNNs](https://arxiv.org/abs/2202.06658)

by Hao Guo, Jiyong Jin and Bin Liu.

# Usage
The code in this repository implements Parsimonious Fast Geometric Ensembling (PFGE), with examples on the CIFAR10 and CIFAR100 datasets.

## Model Training
You can train model using the following command

```
python3 train.py --dir=<DIR> --dataset=<Dataset> --data_path=<path> --model=<Model> --epochs=<Epochs> --lr_init=<lr> \
                 --wd=<Wd> 
```

Parameters：\
```DIR``` — path to training directory where checkpoints will be stored \
```Dataset``` —  dataset name (default: CIFAR10) \
```path``` — path to the data directory \
```Model``` — DNN model name: VGG16, PreResNet164 and WideResNet28x10 \
```Epochs``` — number of training epochs \
```lr``` — initial learning rate \
```Wd``` — weight decay
## Parsimonious Fast Geometric Ensembling (PFGE)
In order to run PFGE you need to pre-train the network to initialize the procedure. Then, you can run PFGE with the following command

```
python3 pfge.py --dir=<DIR> --dataset=<Dataset> --data_path=<path> --model=<Model> --epochs=<Epochs> --lr_init=<lr> \
                --wd=<Wd> --ckpt=<CKPT> --lr_max=<lr1> --lr_min=<lr2> --cycle=<Cycle> --P=<P>  
```

Parameters: \
```CKPT``` — path to the checkpoint saved by ```train.py``` \
```lr1``` — maximum learning rates in the cycle \
```lr2``` — minimum learning rates in the cycle \
```cycle``` — cycle length in epochs \
```P``` —  model recording period
