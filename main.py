'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import torchvision
import torchvision.transforms as transforms
import os
from opts import *
from inference import *
from createDataset import *
from specials import *
from entropyloss import *
from stats import *
from models import *
import numpy as np
from scipy.special import softmax
import wandb
import random


# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# os.environ["WANDB_MODE"] = "dryrun"

def prepData(use_mnist: bool):
    if use_mnist:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        return transform_train, transform_test

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test


def main():
    args = parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train, transform_test = prepData(args.mnist)

    net = chooseNet(args, device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        address = findAddress(args)
        checkpoint = torch.load(address)
        new_weights = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
        net.load_state_dict(new_weights)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    args.union = args.union if not (args.union and args.directTrans) else False

    # Loading dataset
    if args.mnist:
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    if args.rand:
        # dict = np.random.randint(10, size=10)
        dict = np.array([4, 3, 7, 2, 6, 1, 0, 8, 5, 9])
        trainset = changeTar(trainset, dict)
        testset = changeTar(testset, dict)
    if args.part:
        trainset = partialDataset(trainset, num=2000)
        testset = partialDataset(trainset, num=1)

    classessel = random.sample(range(10), 5)
    selectedClasses = [0, 1, 2, 3, 4]
    # selectedClasses = classessel

    print("Selected class are %d", selectedClasses)
    classidx_to_keep = np.array(selectedClasses)

    trainloader1, trainloader2nd = createTrainloader(args, trainset, classidx_to_keep)
    testloader, fullTestset = createTestloader(args, testset, classidx_to_keep)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    if args.ph1:
        args.overclass = False

    ent = EntropyLoss(args, device)
    if args.overclass:
        args.lr = 1e-2

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    if args.directTrans:
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                               weight_decay=5e-4)
        # weight_decay=.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    reduce = 0
    # reduce = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    # reduce =  optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # reduce =  optim.lr_scheduler.StepLR(optimizer,1, gamma=.9)

    # Default values for hyper-parameters we're going to sweep over
    config_defaults = configWand(args)
    config_defaults["Kunif"] = args.Kunif
    config_defaults["Kuniq"] = args.Kuniq
    run = wandb.init(config=config_defaults, reinit=True)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    infer = inference(args, device, net,
                      optimizer, trainloader1,
                      testloader, fullTestset.targets[:],
                      classes, reduce,
                      trainloader2nd, ent)

    if infer.trans and infer.directTrans:
        infer.setTrans()

    # Epoch loop
    for epoch in range(start_epoch, start_epoch + 200):

        spc = specials(args, device, net, trainset, testset, classidx_to_keep, trainloader1, testloader,
                       infer.address)
        spc.flow()
        if args.train:
            infer.train(epoch, False)
        # reduce.step(np.mean(lossCE))
        else:
            if args.train == 0:
                acc = infer.test(epoch, True)
                break
            else:
                acc = infer.test(epoch, False)

        # Early Stopping
        if not (args.mnist) and acc > 95:
            acc = infer.test(epoch, True)
            break

        if acc > 98:
            if args.overclass:
                acc = infer.test(epoch, True)
                break
            else:
                acc = infer.test(epoch, True)
                break

        if args.trans and args.overclass == 0 and acc > 95 and args.resume == 0:
            acc = infer.test(epoch, True)
            break
        if args.trans and args.overclass == 1 and acc > 93.5 and args.resume == 0:
            acc = infer.test(epoch, True)
            break
        if args.mnist and epoch == 5:
            acc = infer.test(epoch, True)
            break
        scheduler.step()


if __name__ == '__main__':
    main()