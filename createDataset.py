import numpy as np
import torch
import copy
from models import *
from models.AE import BasicAE
import os
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data
import torchvision.transforms as transforms


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


def changeTar(set, dict):
    for i in range(len(set.targets)):
        set.targets[i] = dict[int(set.targets[i])]
    tmp = copy.deepcopy(set.classes)
    for i in range(len(set.classes)):
        set.classes[i] = tmp[dict[i]]
    return set


def createDataset(trainset, classidx_to_keep, isTrain, batchSize, imbalance=False, test=False):
    targets = trainset.targets[:]
    outputSet = copy.deepcopy(trainset)

    # first network data
    idx_to_keep = []
    if test:
        for i in range(len(targets)):
            if targets[i] not in classidx_to_keep:
                outputSet.targets[i] = 9
    else:
        for classKeep in classidx_to_keep:
            idxClass = list(np.where(targets == classKeep)[0])

            if imbalance and classKeep % 2:
                del idxClass[int(0.5 * len(idxClass) - 1):-1]
            idx_to_keep += idxClass
        idx_to_keep.sort()

        outputSet.targets = [outputSet.targets[i] for i in idx_to_keep]
        outputSet.data = [outputSet.data[i] for i in idx_to_keep]
    trainloader = torch.utils.data.DataLoader(
        outputSet, batch_size=batchSize, shuffle=isTrain, num_workers=8)
    return trainloader, outputSet


def partialDataset(trainset, percentage):
    indices = np.arange(len(trainset))
    train_indices, _ = train_test_split(indices, train_size=percentage, stratify=trainset.targets)

    # Warp into Subsets and DataLoaders
    train_dataset = torch.utils.data.Subset(trainset, train_indices)
    return train_dataset


def createTrainloader(args, trainset, classidx_to_keep=0):
    if args.openset or args.ae:
        trainloader, _ = createDataset(trainset, classidx_to_keep, True, args.batch)
    elif args.trans and not(args.union):
        trainloader2nd = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch, shuffle=False, num_workers=8)
        fulltrainSet = copy.deepcopy(trainset)
        trainloader, _ = createDataset(fulltrainSet, classidx_to_keep, True, args.batch)
        # trainloader2nd = trainloader
        return trainloader, trainloader2nd
    else:
        if args.union:
            if args.imbalance:
                _, trainset = createDataset(trainset, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), isTrain=True,
                                            batchSize=1, imbalance=args.imbalance, test=False)

            for i, tar in enumerate(trainset.targets):
                trainset.targets[i] = int(np.floor(tar / args.extraclass))
            for i, tar in enumerate(trainset.classes):
                if i < len(trainset.classes) / 2 - 1:
                    trainset.classes[i] = trainset.classes[2 * i] + ' ' + trainset.classes[2 * i - 1]
                else:
                    trainset.classes = trainset.classes[:int(len(trainset.classes) / 2)]
                    break
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch, shuffle=False, num_workers=8)
        classidx_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return trainloader, None


def createTestloader(args, testset, classidx_to_keep=0):
    fullTestset = copy.deepcopy(testset)
    if args.oTest:
        testloader, set = createDataset(testset, classidx_to_keep, False, args.batch)
        fullTestset = set

    elif args.trans and not (args.directTrans) and not(args.union):
        testloader, _ = createDataset(testset, classidx_to_keep, True, args.batch)

    else:
        if args.union:
            fullTestset = copy.deepcopy(testset)
            if args.imbalance:
                _, testset = createDataset(testset, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), isTrain=False,
                                           batchSize=1, imbalance=args.imbalance, test=False)
            for i, tar in enumerate(testset.targets):
                testset.targets[i] = int(np.floor(tar / args.extraclass))

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch, shuffle=False, num_workers=8)
    return testloader, fullTestset


def chooseNet(args, device, extraClasses=2):
    print('==> Building model..')

    # working seeds: 1
    torch.manual_seed(4)
    d = args.mnist + 3 * (args.mnist == 0)
    if args.extraLayer:
        print("extraLayer")
        net = ResNet18(num_classes=5, extraLayer=2, extraClasses=1 + (
                extraClasses - 1) * args.overclass, d=d)  # extraClasses=1 + (extraClasses - 1) * args.overclass)
    elif args.ae:
        net = BasicAE()
    elif args.union and args.overclass:
        net = ResNet18(num_classes=10, d=d)  # extraClasses=1 + (extraClasses - 1) * args.overclass)
    elif args.union:
        net = ResNet18(num_classes=10, d=d)
    elif args.overclass and args.extraLayer:
        net = ResNet18(num_classes=10, extraClasses=1 + (extraClasses - 1) * args.overclass, d=d)
    elif args.overclass and args.featperclass and not args.trans:
        net = ResNet18(num_classes=10, extraClasses=1 + (extraClasses - 1) * args.overclass, extraFeat=True,
                       pool=args.pool, d=d)

    else:
        net = ResNet18(
            num_classes=10 + 10 * (args.extraclass - 1) * args.overclass * (args.trans == 0) * (args.openset == 0),
            pool=args.pool, d=d)
    net = net.to(device)
    return net


def findAddress(args):
    # Load checkpoint.
    if args.mnist:
        address = 'mnist/'
    else:
        address = ''
    if args.extraLayer:
        print("Loading extraLayer")
        assert os.path.isdir('extraLayer/checkpoint'), 'Error: no checkpoint directory found!'
        address += 'extraLayer/checkpoint/ckpt.pth'
    elif args.ae:
        print("Loading autoencoder")
        assert os.path.isdir('autoencoder'), 'Error: no checkpoint directory found!'
        address += 'autoencoder/ckpt.pth'

    elif args.ph2 or args.ph1:
        print("loading ph1 for ph2")
        address += 'ph1/checkpoint/ckpt.pth'
    elif args.extraclass > 2:
        print("Loading 5 overclass")
        address += '5overclass/ckpt.pth'
    elif args.trans and args.overclass and args.featperclass:
        print("Loading trans overclass")
        address += 'featpertrans/ckpt.pth'
    elif args.trans and args.overclass and args.union:
        print("Loading union and overclass")
        address += 'unionOC/checkpoint/ckpt.pth'
    elif args.trans and not args.overclass and args.union:
        print("Loading union class")
        address += 'union/checkpoint/ckpt.pth'
    elif args.trans and args.overclass:
        print("Loading trans overclass")
        address += 'transover/ckpt.pth'
    elif args.trans:
        print("Loading transgender")
        address += 'transfer/ckpt.pth'
    elif args.imbalance:
        print("Loading imbalance")
        address += 'imbalance/checkpoint/ckpt.pth'
    elif args.union and args.overclass:
        print("Loading union and overclass")
        address += 'unionOC/checkpoint/ckpt.pth'

    elif args.union and not args.overclass:
        print("Loading union class")
        address += 'union/checkpoint/ckpt.pth'
    elif args.overclass and args.featperclass:
        print("Loading feat perclass")
        address += 'featperclass/checkpoint/ckpt.pth'
    elif args.overclass and args.openset:
        print("Loading OC openset")
        address += 'OCopenset/checkpoint/ckpt.pth'
    elif args.openset:
        print("Loading openset")
        address += 'openset/checkpoint/ckpt.pth'
    elif args.overclass:
        print("Loading OverClass")
        address += 'overclass/checkpoint/ckpt.pth'

    else:
        address += 'checkpoint/ckpt.pth'
    print(address)
    assert os.path.exists(address), 'Error: no checkpoint directory found!'

    return address


def createFolders(args):
    if args.mnist:
        address = 'mnist/'
    else:
        address = ''
    if args.extraLayer:
        os.makedirs('extraLayer', exist_ok=True )
        os.makedirs('extraLayer/checkpoint', exist_ok=True)
        address += 'extraLayer/checkpoint/ckpt.pth'
    if args.ae:
        os.makedirs('autoencoder', exist_ok=True)
        address += 'autoencoder/ckpt.pth'
    elif args.ph1:
        os.makedirs('ph1', exist_ok=True )
        os.makedirs('ph1/checkpoint', exist_ok=True)
        address += 'ph1/checkpoint/ckpt.pth'

    elif args.ph2:
        os.makedirs('ph2', exist_ok=True )
        os.makedirs('ph2/checkpoint', exist_ok=True)
        address += 'ph2/checkpoint/ckpt.pth'
    elif args.extraclass > 2:
        os.makedirs('5overclass', exist_ok=True )
        os.makedirs('5overclass/checkpoint', exist_ok=True)
        address += '5overclass/checkpoint/ckpt.pth'
    elif args.trans and args.overclass and args.featperclass:
        os.makedirs('featpertrans', exist_ok=True)
        os.makedirs('featpertrans/checkpoint', exist_ok=True)
        address += 'featpertrans/checkpoint/ckpt.pth'
    elif args.trans and args.overclass:
        os.makedirs('transover', exist_ok=True)
        os.makedirs('transover/checkpoint', exist_ok=True)
        address += 'transover/checkpoint/ckpt.pth'
    elif args.trans:
        os.makedirs('transfer', exist_ok=True)
        os.makedirs('transfer/checkpoint', exist_ok=True)
        address += 'transfer/checkpoint/ckpt.pth'
    elif args.imbalance:
        os.makedirs('imbalance', exist_ok=True)
        os.makedirs('imbalance/checkpoint', exist_ok=True)
        address += 'imbalance/checkpoint/ckpt.pth'
    elif args.union and args.overclass:
        os.makedirs('unionOC', exist_ok=True)
        os.makedirs('unionOC/checkpoint', exist_ok=True)
        address += 'unionOC/checkpoint/ckpt.pth'
    elif args.union and not args.overclass:
        os.makedirs('union', exist_ok=True)
        os.makedirs('union/checkpoint', exist_ok=True)
        address += 'union/checkpoint/ckpt.pth'
    elif args.overclass and args.featperclass:
        os.makedirs('featperclass', exist_ok=True)
        os.makedirs('featperclass/checkpoint', exist_ok=True)
        address += 'featperclass/checkpoint/ckpt.pth'
    elif args.overclass and args.openset:
        os.makedirs('OCopenset', exist_ok=True)
        os.makedirs('OCopenset/checkpoint', exist_ok=True)
        address += 'OCopenset/checkpoint/ckpt.pth'
    elif args.openset:
        os.makedirs('openset', exist_ok=True)
        os.makedirs('openset/checkpoint', exist_ok=True)
        address += 'openset/checkpoint/ckpt.pth'
    elif args.overclass:
        os.makedirs('overclass', exist_ok=True)
        os.makedirs('overclass/checkpoint', exist_ok=True)
        address += 'overclass/checkpoint/ckpt.pth'
    else:
        os.makedirs('checkpoint', exist_ok=True)
        address += 'checkpoint/ckpt.pth'
    return address


def weight_lst(net):
    """
        :param self.
        :return: A list of iterators of the network parameters.
    """
    return [w for w in net.parameters()]
