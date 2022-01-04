import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mnist', '-mnist', default=0, action='store_true')

    args = parser.parse_args()
    return args
args = parse()

if args.mnist: #mnist
    os.system('python main.py -t -mnist')
    os.system('python main.py -t  -union -mnist ')
    os.system('python main.py -t   -union -overclass -mnist')
    os.system('python main.py -t    -overclass -mnist')


os.system('python main.py -t --batch 300')

os.system('python main.py -t  -union -overclass --batch 300')

os.system('python main.py -t  -union --batch 300')

# os.system('python main.py -t  -overclass --extraclass 5')