### (1) augmentation baselines

python3 train.py -c -w -d cifar10 -e 300 -n presnet50 
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr randaug
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr autoaug
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr augmix
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr trivaug
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr mixup
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr cutmix

python3 train.py -c -w -d cifar100 -e 300 -n presnet50 
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr randaug
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr autoaug
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr augmix
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr mixup
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr cutmix

python3 train.py -c -w -d tin -e 300 -n presnet50 
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr randaug
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr autoaug
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr augmix
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr mixup
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr cutmix

### (2) Mollification runs, using (xnb,beta=2,pow=1)

python3 train.py -c -w -d cifar10 -e 300 -n presnet50                   -m
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr         -m
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr randaug -m
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr autoaug -m
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr augmix  -m
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr trivaug -m
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr mixup   -m
python3 train.py -c -w -d cifar10 -e 300 -n presnet50 --aug fcr cutmix  -m

python3 train.py -c -w -d cifar100 -e 300 -n presnet50                   -m
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr         -m
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr randaug -m
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr autoaug -m
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr augmix  -m
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr mixup   -m
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr cutmix  -m

python3 train.py -c -w -d tin -e 300 -n presnet50                   -m
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr         -m
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr randaug -m
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr autoaug -m
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr augmix  -m
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr mixup   -m
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr cutmix  -m

### (3) ablation: n/b only 

python3 train.py -c -w -d cifar10  -e 300 -n presnet50 --aug fcr trivaug -m --batch xn
python3 train.py -c -w -d cifar10  -e 300 -n presnet50 --aug fcr trivaug -m --batch xb
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --batch xn
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --batch xb
python3 train.py -c -w -d tin      -e 300 -n presnet50 --aug fcr trivaug -m --batch xn
python3 train.py -c -w -d tin      -e 300 -n presnet50 --aug fcr trivaug -m --batch xb

### (4) ablation: betas

python3 train.py -c -w -d cifar10  -e 300 -n presnet50 --aug fcr trivaug -m --beta 2
python3 train.py -c -w -d cifar10  -e 300 -n presnet50 --aug fcr trivaug -m --beta 4
python3 train.py -c -w -d cifar10  -e 300 -n presnet50 --aug fcr trivaug -m --beta 9
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --beta 2
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --beta 4
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --beta 9
python3 train.py -c -w -d tin      -e 300 -n presnet50 --aug fcr trivaug -m --beta 2
python3 train.py -c -w -d tin      -e 300 -n presnet50 --aug fcr trivaug -m --beta 4
python3 train.py -c -w -d tin      -e 300 -n presnet50 --aug fcr trivaug -m --beta 9

# ### (5) ablation: no label smoothing

python3 train.py -c -w -d cifar10  -e 300 -n presnet50 --aug fcr trivaug -m --npow 100 --bpow 100
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 100 --bpow 100
python3 train.py -c -w -d tin      -e 300 -n presnet50 --aug fcr trivaug -m --npow 100 --bpow 100

### (6) ablation: pows

python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 0.5 --bpow 0.5
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 0.5 --bpow 1
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 0.5 --bpow 2
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 1 --bpow 0.5
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 1 --bpow 2
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 2 --bpow 0.5
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 2 --bpow 1
python3 train.py -c -w -d cifar100 -e 300 -n presnet50 --aug fcr trivaug -m --npow 2 --bpow 2

python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 0.5 --bpow 0.5
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 0.5 --bpow 1
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 0.5 --bpow 2
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 1 --bpow 0.5
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 1 --bpow 2
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 2 --bpow 0.5
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 2 --bpow 1
python3 train.py -c -w -d tin -e 300 -n presnet50 --aug fcr trivaug -m --npow 2 --bpow 2
