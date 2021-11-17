import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mnist', '-mnist', default=0, action='store_true')

    args = parser.parse_args()
    return args
args = parse()

if args.mnist: #mnist
    os.system('python main.py -t ')
    os.system('python main.py -t  -union ')
    os.system('python main.py -t   -union -overclass')
    os.system('python main.py -t    -overclass')


os.system('python main.py -t -rand')

os.system('python main.py -t  -rand -union -overclass')

os.system('python main.py -t  -union -rand')

os.system('python main.py -t  -overclass --extraclass 5')