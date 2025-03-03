# Robust Classification by Coupling Data Mollification with Label Smoothing

Code for paper [Robust Classification by Coupling Data Mollification with Label Smoothing](https://arxiv.org/abs/2406.01494) (AISTATS 2025)

Implementation of supervised mollification: couple image noising with label smoothing (AISTATS 2025) 

Repo supports CIFAR-10, CIFAR-100, TinyImagenet and various network architectures.

To reproduce paper results:
- Edit `config` file to add wandb details 
- Download CIFAR10/100/TinyImagenet (see `/data`), and run `data/process-datasets.py`. The code `src/data.py` expects datasets to be .npy files with (N,C,W,H) array inside
– Execute `runs_paper.sh` on a GPU cluster 

The code depends on
- pytorch
- lightning
- wandb
- omegaconf
- torch_dct

Use `train.py --help` to train models

Try example runs:

- `python3 train.py -d tin -n presnet50 --aug fcr trivaug`     (without mollification)
- `python3 train.py -d tin -n presnet50 --aug fcr trivaug -m`     (with mollification)

We obtain approximately errors of 
 - 5% CIFAR-10
 - 20% CIFAR-100
 - 32% TIN
