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

# print (str(datetime.datetime.now()))
# print("10^-3")
# os.system('python main.py -t  -time --lamda 1e-3')
# print (str(datetime.datetime.now()))
#
# print("510^-3")
# os.system('python main.py -t  -time --lamda 5e-3')
# print("10^-2")
# os.system('python main.py -t  -time --lamda 1e-2')
# print("510^-1")
# os.system('python main.py -t  -time --lamda 5e-1')
# os.system('python main.py -t --batch 300')
#
# os.system('python main.py -t  -union -overclass --batch 300')
#
# os.system('python main.py -t  -union --batch 300')

# os.system('python main.py -t  -overclass --extraclass 5')
# os.system('python main.py -t -time -f  -openset  -kl --lamda 0.001 --falseLayer 0')
# os.system('python main.py -t -time -f  -openset  -kl --lamda 0.001 --falseLayer 1 -fc')
# os.system('python main.py -t --save_path "newshit/mu 0.001" -f  -openset  -kl --lamda 0.001 --falseLayer 0 -fc --mu 0.001')
# os.system('python main.py -t --save_path "newshit/mu 0.005" -f  -openset  -kl --lamda 0.001 --falseLayer 0 -fc --mu 0.005')
# os.system('python main.py -t --save_path "newshit/mu 0.05" -f  -openset  -kl --lamda 0.001 --falseLayer 0 -fc --mu 0.05')
# os.system('python main.py -t --save_path "newshit/mu 0.1" -f  -openset  -kl --lamda 0.001 --falseLayer 0 -fc --mu 0.1')
# os.system('python main.py -t --save_path "newshit/mu 0.3" -f  -openset  -kl --lamda 0.001 --falseLayer 0 -fc --mu 0.3')
# os.system('python main.py -t --save_path "newshit/mu 0.5" -f  -openset  -kl --lamda 0.001 --falseLayer 0 -fc --mu 0.5')

# os.system('python main.py -t --save_path "newshit3/mu 0.001" -f  -openset  -kl --falseLayer 0 -fc --mu 0.001')

# os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.001 L-3 no reg" -f -entropy -openset 1 -kl --falseLayer 1 -fc --mu 0.001')
# # os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.01 lamda 0.001 L-3" --lamda 0.001 -f -entropy -openset  -kl --falseLayer 1 -fc --mu 0.01')
# # os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.1 lamda 0.001 L-3" -f --lamda 0.001  -entropy -openset  -kl  --falseLayer 1 -fc --mu 0.1')
# #
# os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.001 L-4 no reg " -f -entropy -openset  -kl --falseLayer 2 -fc --mu 0.001')
# # os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.01 lamda 0.001 L-4" --lamda 0.001 -f -entropy -openset  -kl --falseLayer 2 -fc --mu 0.01')
# # os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.1 lamda 0.001 L-4" -f --lamda 0.001  -entropy -openset  -kl  --falseLayer 2 -fc --mu 0.1')
# #
# os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.001 L-5 no reg" -f -entropy -openset  -kl --falseLayer 3 -fc --mu 0.001')
# os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.01 lamda 0.001 L-5" --lamda 0.001 -f -entropy -openset  -kl --falseLayer 3 -fc --mu 0.01')
# os.system('python main.py -t --save_path "entorpy no bias through layers/mu 0.1 lamda 0.001 L-5" -f --lamda 0.001  -entropy -openset  -kl  --falseLayer 3 -fc --mu 0.1')

# os.system('python main.py -t --save_path "newshit3 - False CE/mu 0.01" -f  -openset  -kl --falseLayer 0 -fc --mu 0.01')
# os.system('python main.py -t --save_path "newshit3 - False CE/mu 0.1" -f  -openset  -kl  --falseLayer 0 -fc --mu 0.1')
# os.system('python main.py -t --save_path "newshit3 - False CE/mu 0.4" -f  -openset  -kl  --falseLayer 0 -fc --mu 0.4')
# os.system('python main.py -t --save_path "newshit3 - False CE/mu 0.01 lamda 0.001" --lamda 0.001 -f  -openset  -kl --falseLayer 0 -fc --mu 0.01')
# os.system('python main.py -t --save_path "newshit3 - False CE/mu 0.1 lamda 0.001" -f --lamda 0.001 -openset  -kl  --falseLayer 0 -fc --mu 0.1')
# os.system('python main.py -t --save_path "newshit3 - False CE/mu 0.4 lamda 0.001" -f --lamda 0.001 -openset  -kl  --falseLayer 0 -fc --mu 0.4')

# false layer 1

# os.system('python main.py -t  -f --save_path "newYear/layer L-3 mu 0.001" -openset  -kl --lamda 0.001 --falseLayer 1 -fc --mu 0.001')
# os.system('python main.py -t  -f --save_path "newYear/layer L-3 mu 0.001" -openset  -kl --lamda 0.004 --falseLayer 1 -fc --mu 0.001')
#
# # false layer 2
#
# os.system('python main.py -t  -f --save_path "newYear/layer L-4 mu 0.001" -openset  -kl --lamda 0.001 --falseLayer 2 -fc --mu 0.001')
# os.system('python main.py -t  -f --save_path "newYear/layer L-4 mu 0.001" -openset  -kl --lamda 0.004 --falseLayer 2 -fc --mu 0.001')
#
# # false layer 3
#
# os.system('python main.py -t  -f --save_path "newYear/layer L-5 mu 0.001" -openset  -kl --lamda 0.001 --falseLayer 3 -fc --mu 0.001')
# os.system('python main.py -t  -f --save_path "newYear/layer L-5 mu 0.001" -openset  -kl --lamda 0.004 --falseLayer 3 -fc --mu 0.001')

# os.system('python main.py -t --save_path "newoverclass/1reg 0.0004" -f  -openset -overclass --Kuniq 0.1 --Kunif 1 --lamda 0.0004' )
# os.system('python main.py -t --save_path "newoverclass/1imbalnce reg 0.0004" -f  -union -overclass --Kuniq 0.1 --Kunif 1 --lamda 0.0004 -imbalance' )
# os.system('python main.py -t --save_path "newoverclass/2reg 0.0004" -f  -openset -overclass --Kuniq 0.1 --Kunif 2 --lamda 0.0004' )
# os.system('python main.py -t --save_path "newoverclass/2imbalnce reg 0.0004" -f  -union -overclass --Kuniq 0.1 --Kunif 2 --lamda 0.0004 -imbalance' )
# os.system('python main.py -t --save_path "newoverclass/3reg 0.0004" -f  -openset -overclass --Kuniq 0.1 --Kunif 2.5 --lamda 0.0004' )
os.system('python main.py -t --save_path "newoverclass/3imbalnce reg 0.0004" -f  -union -overclass --Kuniq 0.1 --Kunif 2.5 --lamda 0.0004 -imbalance' )

# os.system('python main.py -t --save_path "newoverclass/1reg 0.0003" -f  -openset -overclass --Kuniq 0.1 --Kunif 1 --lamda 0.0003' )
os.system('python main.py -t --save_path "newoverclass/1imbalnce reg 0.0003" -f  -union -overclass --Kuniq 0.1 --Kunif 1 --lamda 0.0003 -imbalance' )
os.system('python main.py -t --save_path "newoverclass/2reg 0.0003" -f  -openset -overclass --Kuniq 0.1 --Kunif 2 --lamda 0.0003' )
os.system('python main.py -t --save_path "newoverclass/2imbalnce reg 0.0003" -f  -union -overclass --Kuniq 0.1 --Kunif 2 --lamda 0.0003 -imbalance' )
os.system('python main.py -t --save_path "newoverclass/3reg 0.0003" -f  -openset -overclass --Kuniq 0.1 --Kunif 2.5 --lamda 0.0003' )
os.system('python main.py -t --save_path "newoverclass/3imbalnce reg 0.0003" -f  -union -overclass --Kuniq 0.1 --Kunif 2.5 --lamda 0.0003 -imbalance' )

