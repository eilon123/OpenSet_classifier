'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import torchvision
import os
from opts import *
from Inference import *
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
    if args.ae:
        selectedClasses = [0,1]

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
        # args.lr = 1e-2
        x=2
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    if args.directTrans:
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                               weight_decay=5e-4)

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
    if args.resume == 0:
        best_acc = 0
        if args.ae:
            best_acc = np.inf

    infer = Inference(args, device, net,
                      optimizer, trainloader1,
                      testloader, fullTestset.targets[:],
                      classes, reduce,
                      trainloader2nd, ent,best_acc)

    if infer.trans and infer.directTrans:
        infer.setTrans()
    infer.setclasses(selectedClasses)
    # Epoch loop
    if args.ae ==0:
        w = weight_lst(net)
        weightValue = torch.sort(w[-2])[0]

        x = torch.sort(w[-2])[1]
        highest = []
        lowest = []
        wlowest = []
        whighest = []
        for i in range(int((10 + args.overclass * 10 )/(1+args.openset))):
            lowest.append(x[i][0:10])
            highest.append(x[i][-11:-1])
            wlowest.append(weightValue[i][0:10])
            whighest.append(weightValue[i][-11:-1])
        from statistics import mean

        print("highest mean",torch.mean(torch.cat(whighest,0)).item())
        print("highest var",torch.var(torch.cat(whighest,0)).item())

        print("lowest mean",torch.mean(torch.cat(wlowest,0)).item())
        print("lowest var", torch.var(torch.cat(wlowest,0)).item())
        infer.setWeights(highest,lowest)
    for epoch in range(start_epoch, start_epoch + 200):

        spc = specials(args, device, net, trainset, testset, classidx_to_keep, trainloader1, testloader,
                       infer.address)
        # spc.flow()
        if args.train:
            if args.ae:
                infer.AEtrain(epoch)
            else:
                infer.train(epoch, False)
        # reduce.step(np.mean(lossCE))

        if args.train == 0:
            if args.ae:
                acc = infer.AEtest(epoch, True)
            else:
                acc = infer.test(epoch, True)
            break
        else:
            print("epoch num ",epoch)
            if epoch%4 == 0:
                if args.ae:
                    acc = infer.AEtest(epoch,False)
                else:

                    acc = infer.test(epoch, False)
            if epoch == 50 and args.ae:
                acc = infer.AEtest(epoch, True)

        if args.ae:
            if acc < best_acc:
                best_acc = acc
        else:
            if acc > best_acc:
                best_acc = acc

        print("================= best acc is",best_acc)
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
        if args.overclass and args.openset and acc > 46:
            acc = infer.test(epoch, True)
            break
        if args.trans and args.overclass == 0 and acc > 95 and args.resume == 0:
            acc = infer.test(epoch, True)
            break
        if args.overclass and acc >80:
            dbg = 3
        if args.trans and args.overclass == 1 and acc > 91 and args.resume == 0:
            acc = infer.test(epoch, True)
            break
        if args.mnist and epoch == 5:
            acc = infer.test(epoch, True)
            break
        scheduler.step()


if __name__ == '__main__':
    main()
