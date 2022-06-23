import os
import argparse
import time
import datetime
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

print (str(datetime.datetime.now()))
print("10^-3")
os.system('python main.py -t  -time --lamda 1e-3')
print (str(datetime.datetime.now()))

print("510^-3")
os.system('python main.py -t  -time --lamda 5e-3')
print("10^-2")
os.system('python main.py -t  -time --lamda 1e-2')
print("510^-1")
os.system('python main.py -t  -time --lamda 5e-1')
# os.system('python main.py -t --batch 300')
#
# os.system('python main.py -t  -union -overclass --batch 300')
#
# os.system('python main.py -t  -union --batch 300')

# os.system('python main.py -t  -overclass --extraclass 5')