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
from pca_util import *
from opts import *
from inference import *
from createDataset import *
from nearestNeighbour import *
from entropyloss import *
from stats import *
from Osvm import *
from models import *
import numpy as np
from scipy.special import softmax
import wandb
import random
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# os.environ["WANDB_MODE"] = "dryrun"

args = parse()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.mnist:

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
else:

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

net = chooseNet(args, device)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    address = findAddress(args)
    checkpoint = torch.load(address)
    new_weights = {k.replace('module.',''):v for k,v in checkpoint['net'].items()}
    net.load_state_dict(new_weights)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
if args.reserve:
    print("Loading reserve")
    assert os.path.isdir('reserve/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('reserve/checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.union and args.directTrans:
    args.union = False
if args.mnist:
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)
else:
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
if args.rand:
    # dict = np.random.randint(10, size=10)
    dict = np.array([4,3,7,2,6,1,0,8,5,9])
    trainset = changeTar(trainset,dict)
if args.part:
    trainset = partialDataset(trainset, num=2000)
classessel = random.sample(range(10), 5)
selectedClasses = [0, 1, 2, 3, 4]
# selectedClasses = classessel
print("selected class are %d", selectedClasses)
classidx_to_keep = np.array(selectedClasses)
trainloader1, trainloader2nd = createTrainloader(args, trainset, classidx_to_keep)


if args.mnist:
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)
else:

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
if args.rand:


    testset = changeTar(testset,dict)
if args.part:
    testset = partialDataset(trainset, num=1)
testloader, fullTestset = createTestloader(args, testset,classidx_to_keep)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model

if args.ph1:
    args.overclass = False

if args.overclass:
    args.lr = 1e-2
if args.overclass:
    ent = EntropyLoss(args,device)
m = nn.LogSoftmax(dim=1)
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
i=2


# Default values for hyper-parameters we're going to sweep over
config_defaults = configWand(args)
config_defaults["Kunif"] = args.Kunif
config_defaults["Kuniq"] = args.Kuniq
run = wandb.init(config=config_defaults, reinit=True)

# Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config
if args.overclass:
    ent = EntropyLoss(args, device)
inf = inference(args ,device,net,optimizer,trainloader1,testloader,fullTestset.targets[:],classes,reduce,trainloader2nd,ent)
if inf.trans and inf.directTrans:
    inf.setTrans()
for epoch in range(start_epoch, start_epoch + 200):

    if args.train:
        inf.train(epoch, False)

    # reduce.step(np.mean(lossCE))
    if args.NN:
        centroids = getCentroids(net, device, trainset, classidx_to_keep)
        newTestset = copy.deepcopy(trainset)
        newTestset.targets = torch.utils.data.ConcatDataset([trainset.targets, testset.targets])
        newTestset.data = torch.utils.data.ConcatDataset([trainset.data, testset.data])

        tsne, TSNEcenteroids = getCentroidsTSNE(net, device, newTestset, classidx_to_keep)
        correctCent = classify(net, device, testloader, centroids)
        correcttsne = classifyTSNE(tsne, testloader, TSNEcenteroids, len(trainset), device)
        print("centriod correct is: ", correctCent)
        print("tsne correct is: ", correcttsne)

        break
    if args.osvm:

        cls = osvm(args, device, net, trainset, testloader, testset.targets[:], inf.address)
        #overfit
        # cls = osvm(args, device, net, trainset, trainloader1, trainset.targets[:],inf.address)
        set = trainset
        if args.overclass:
            set = cls.createoverClassDataSet()

        # testloader,testset = createDataset( testset,np.array([0,1,2]),isTrain=False, batchSize=args.batch,
        #                                test=False)
        # cls.setTestloader(testloader,testset.targets)
        clf = cls.getSupport(set)
        cls.testSVM(clf)

        break
    if args.deepNN:
        centroidsFeat, centroidsL1, centroidsL2, centroidsL3, centroidsL4 = getCentroids(net, device, trainset,
                                                                                             classidx_to_keep)
        classify(net, device, testloader, centroidsFeat, centroidsL1, centroidsL2, centroidsL3, centroidsL4)
    if args.pca:
        pca_t = pca_util(args ,device,net,trainloader1,testloader)
        pca_t.eval()
        pca_t.evalLambda()
        break
    if args.level:
        cls = osvm(args, device, net, trainset, testloader, testset.targets[:], inf.address)
        set = cls.createoverClassDataSet()
        if args.overclass and args.trans or not(args.overclass):
            classidx_to_keep = np.array([0,1,2,3,4,5,6,7,8,9])
        elif args.overclass:
            classidx_to_keep = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19])
        centroids ,clusterVar= getCentroids(net, device, set, classidx_to_keep)
        disth = np.zeros(shape=(20, 20))
        x = np.sqrt(clusterVar)
        var = np.sum(x, axis=1)
        print("var is: ", var)
        for row, cent in enumerate(centroids):
            if args.overclass and row%2:
                continue
            for col, _ in enumerate(centroids):
                disth[row, col] = (LA.norm(cent - centroids[col], ord=2))
                # tmp = np.delete(disth[row,:],row)
            if args.overclass:
                print("dist to sub class is ", disth[row, row + 1])
                tmp = np.delete(disth[row, :], row + 1)

            tmp = disth[disth != 0]

            print("avg dist to classes is",np.average(tmp))
        print(disth)
        break
    elif args.orth:
        centroids, _, _, _, _ = getCentroids(net, device, trainset, classidx_to_keep)
        hist = gethistproduct(net, device, trainloader1, centroids)
    else:
        if args.train == 0:
            acc = inf.test(epoch, True)
            break
        else:
            acc = inf.test(epoch, False)
    if not(args.mnist) and acc > 95:
        acc = inf.test(epoch, True)
        break

    if acc > 98 and args.overclass:

        acc = inf.test(epoch, True)
        break

    if acc > 98 and not(args.overclass):
        acc = inf.test(epoch, True)
        break

    if args.trans  and args.overclass == 0 and acc > 95 and args.resume == 0:
        acc = inf.test(epoch, True)
        break
    if args.trans  and args.overclass == 1 and acc > 93.5 and args.resume == 0:
        acc = inf.test(epoch, True)
        break
    if args.mnist and epoch ==5:
        acc = inf.test(epoch, True)
        break
    scheduler.step()
        # if epoch > 30 and acc < 25:
        #     break

