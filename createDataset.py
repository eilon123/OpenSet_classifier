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
import datetime

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


def createDataset(trainset, classidx_to_keep, isTrain, batchSize, imbalance=False, test=False,falselabels=0,part=0):
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
    if falselabels:
        N = int(len(outputSet.targets) * part/10)
        for i in range(N):
            outputSet.targets[i] = int(not(outputSet.targets[i]))
    trainloader = torch.utils.data.DataLoader(
        outputSet, batch_size=batchSize, shuffle=isTrain, num_workers=8)
    return trainloader, outputSet


def partialDataset(trainset, percentage):
    indices = np.arange(len(trainset))
    train_indices, _ = train_test_split(indices, train_size=percentage, stratify=trainset.targets)

    # Warp into Subsets and DataLoaders
    train_dataset = torch.utils.data.Subset(trainset, train_indices)
    print(len(train_dataset.dataset.targets))
    return train_dataset


def createTrainloader(args, trainset, classidx_to_keep=0):
    if args.openset and not args.trans or args.ae:
        if args.union and args.overclass:
            _, trainset = createDataset(trainset, np.arange(len(classidx_to_keep)), isTrain=True,
                                        batchSize=1, imbalance=args.imbalance, test=False)
            for i, tar in enumerate(trainset.targets):
                trainset.targets[i] = int(np.floor(tar / args.extraclass))
            for i, tar in enumerate(trainset.classes):
                if i < len(trainset.classes) / 2 - 1:
                    trainset.classes[i] = trainset.classes[2 * i] + ' ' + trainset.classes[2 * i - 1]
                else:
                    trainset.classes = trainset.classes[:int(len(trainset.classes) / 2)]
                    break
            trainloader, _ = createDataset(trainset, np.arange(len(classidx_to_keep)/2), True, args.batch, falselabels=args.falselabels,
                                       part=args.parts)
        else:
            trainloader, _ = createDataset(trainset, classidx_to_keep, True, args.batch,falselabels=args.falselabels,part=args.parts)
    elif args.trans and not(args.union):

        fulltrainSet = copy.deepcopy(trainset)
        trainloader, _ = createDataset(fulltrainSet, np.arange(10), True, args.batch)
        return trainloader
    else:
        if args.union:
            if args.imbalance:
                _, trainset = createDataset(trainset, np.arange(10), isTrain=True,
                                            batchSize=1, imbalance=args.imbalance, test=False)
            # tmp

            # _, trainset = createDataset(trainset, np.array([0,1,2,3]), isTrain=True,
            #                             batchSize=1, test=False)
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
    return trainloader


def createTestloader(args, testset, classidx_to_keep=0,externalotest =0):
    fullTestset = copy.deepcopy(testset)
    if args.openset or externalotest:
        if args.union and args.overclass:
            _, testset = createDataset(testset, np.arange(len(classidx_to_keep)), isTrain=True,
                                       batchSize=1, imbalance=args.imbalance, test=False)
            for i, tar in enumerate(testset.targets):
                testset.targets[i] = int(np.floor(tar / args.extraclass))
            for i, tar in enumerate(testset.classes):
                if i < len(testset.classes) / 2 - 1:
                    testset.classes[i] = testset.classes[2 * i] + ' ' + testset.classes[2 * i - 1]
                else:
                    testset.classes = testset.classes[:int(len(testset.classes) / 2)]
                    break
            testloader, _ = createDataset(testset, np.arange(len(classidx_to_keep) / 2), True, args.batch,
                                           falselabels=args.falselabels,
                                           part=args.parts)
        else:
            testloader, set = createDataset(testset, classidx_to_keep, False, args.batch)
            fullTestset = set


    else:
        if args.union:
            fullTestset = copy.deepcopy(testset)
            # tmp
            # _, testset = createDataset(testset, np.array([0, 1, 2, 3]), isTrain=True,
            #                             batchSize=1, test=False)
            # _,fullTestset= createDataset(testset, np.array([0, 1, 2, 3]), isTrain=True,
            #                             batchSize=1, test=False)
            if args.imbalance:
                _, testset = createDataset(testset, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), isTrain=False,
                                           batchSize=1, imbalance=args.imbalance, test=False)
            fullTestset = copy.deepcopy(testset)
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
    # elif args.ae:
    #     net = BasicAE()
    elif args.union and args.overclass and not args.openset:
        net = ResNet18(num_classes=10, d=d)  # extraClasses=1 + (extraClasses - 1) * args.overclass)
    elif args.union and args.overclass and  args.openset:
        net = ResNet18(num_classes=6 , d=d)  # extraClasses=1 + (extraClasses - 1) * args.overclass)
    elif args.union:
        net = ResNet18(num_classes=10, d=d)
    elif args.overclass and args.extraLayer:
        net = ResNet18(num_classes=10, extraClasses=1 + (extraClasses - 1) * args.overclass, d=d)
    elif args.overclass and args.featperclass and not args.trans:
        net = ResNet18(num_classes=10, extraClasses=1 + (extraClasses - 1) * args.overclass, extraFeat=True,
                       pool=args.pool, d=d,kl=args.kl)

    else:
        net = ResNet18(
            num_classes=10  +2*args.openset*args.overclass +10 * (args.extraclass - 1) * args.overclass * (args.trans == 0) * (args.openset == 0),
             d=d,kl=args.kl)
        # net = ResNet34(
        #     num_classes=10 + 2 * args.openset * args.overclass + 10 * (args.extraclass - 1) * args.overclass * (
        #                 args.trans == 0) * (args.openset == 0),
        #      d=d, kl=args.kl)
    net = net.to(device)
    return net


def findAddress(args):
    # Load checkpoint.
    if args.mnist:
        address = 'mnist/'
    else:
        address = ''
    if len(args.load_path) >0:
        print("Loading special address")
        assert os.path.isdir(args.load_path + '/checkpoint'), 'Error: no checkpoint directory found!'
        address += args.load_path + '/'


    # elif args.ae:
    #     print("Loading autoencoder")
    #     assert os.path.isdir('autoencoder'), 'Error: no checkpoint directory found!'
    #     address += 'autoencoder/ckpt.pth'


    elif args.extraclass > 2:
        print("Loading 5 overclass")
        address += '5overclass/'
    elif args.trans and args.overclass and args.featperclass:
        print("Loading trans overclass")
        address += 'featpertrans/ckpt.pth'
    elif args.trans and args.overclass and not args.union:
        print("Loading trans overclass")
        address += 'transover/ckpt.pth'
    elif args.trans and not args.overclass and not args.union:
        print("Loading transgender")
        address += 'transfer/ckpt.pth'
    elif args.imbalance:
        print("Loading imbalance")
        address += 'imbalance/checkpoint/ckpt.pth'
    elif args.union and args.overclass:
        print("Loading union and overclass")
        address += 'unionOC/'

    elif args.union and not args.overclass:
        print("Loading union class")
        address += 'union/'
    elif args.overclass and args.featperclass:
        print("Loading feat perclass")
        address += 'featperclass/checkpoint/ckpt.pth'
    elif args.overclass and args.openset:
        print("Loading OC openset")
        address += 'OCopenset/'
    elif args.openset:
        print("Loading openset")
        address += 'openset/'
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
    if len(args.save_path) > 0:
        address = args.save_path
        os.makedirs(address, exist_ok=True)

        os.makedirs(address + '/checkpoint', exist_ok=True)
        assert os.path.isdir(args.save_path + '/checkpoint'), 'Error: no checkpoint directory found!'
        # address += args.save_path + '/checkpoint/ckpt.pth'
        address += '/'
    elif len(args.load_path) > 0:
        address = args.load_path
        os.makedirs(address, exist_ok=True)

        os.makedirs(address + '/checkpoint', exist_ok=True)
        assert os.path.isdir(args.load_path + '/checkpoint'), 'Error: no checkpoint directory found!'
        # address += args.save_path + '/checkpoint/ckpt.pth'
        address += '/'
    elif args.time:
        address += str(datetime.datetime.now())
        os.makedirs(address, exist_ok=True)

        os.makedirs(address + '/checkpoint', exist_ok=True)

        address +=  '/'
                    # 'checkpoint/ckpt.pth'
    elif args.ae:
        os.makedirs('autoencoder', exist_ok=True)
        address += 'autoencoder/ckpt.pth'
    elif args.extraclass > 2:
        os.makedirs('5overclass', exist_ok=True )
        os.makedirs('5overclass/checkpoint', exist_ok=True)
        address += '5overclass/'
    elif args.trans and args.overclass and args.featperclass:
        os.makedirs('featpertrans', exist_ok=True)
        os.makedirs('featpertrans/checkpoint', exist_ok=True)
        address += 'featpertrans/'
    elif args.trans and args.overclass:
        os.makedirs('transover', exist_ok=True)
        os.makedirs('transover/checkpoint', exist_ok=True)
        address += 'transover/'
    elif args.trans:
        os.makedirs('transfer', exist_ok=True)
        os.makedirs('transfer/checkpoint', exist_ok=True)
        address += 'transfer/'
    elif args.imbalance:
        os.makedirs('imbalance', exist_ok=True)
        os.makedirs('imbalance/checkpoint', exist_ok=True)
        address += 'imbalance/'
    elif args.union and args.overclass:
        os.makedirs('unionOC', exist_ok=True)
        os.makedirs('unionOC/checkpoint', exist_ok=True)
        # address += 'unionOC/checkpoint/ckpt.pth'
        address += 'unionOC/'
    elif args.union and not args.overclass:
        os.makedirs('union', exist_ok=True)
        os.makedirs('union/checkpoint', exist_ok=True)
        address += 'union/'
    elif args.overclass and args.featperclass:
        os.makedirs('featperclass', exist_ok=True)
        os.makedirs('featperclass/checkpoint', exist_ok=True)
        address += 'featperclass/'
    elif args.overclass and args.openset:
        os.makedirs('OCopenset', exist_ok=True)
        os.makedirs('OCopenset/checkpoint', exist_ok=True)
        address += 'OCopenset/'
    elif args.openset:
        os.makedirs('openset', exist_ok=True)
        os.makedirs('openset/checkpoint', exist_ok=True)
        address += 'openset/'
    elif args.overclass:
        os.makedirs('overclass', exist_ok=True)
        os.makedirs('overclass/checkpoint', exist_ok=True)
        address += 'overclass/'
    else:
        os.makedirs('checkpoint', exist_ok=True)
        # address += 'checkpoint/ckpt.pth'
    return address


def weight_lst(net):
    """
        :param self.
        :return: A list of iterators of the network parameters.
    """
    return [w for w in net.parameters()]

def bulid_openmax(vectorWise, pca, iskmeans, alpha, orderW, quicknet, net):
    open_max_t = {"vectorWise": 1, "pca": 0, "iskmeans": 0, "alpha": 0.5, "orderW": [], "quicknet": 0, "net": 0}
    open_max_t["vectorWise"] = vectorWise
    open_max_t["pca"] = pca
    open_max_t["iskmeans"] = iskmeans
    open_max_t["alpha"] = alpha
    open_max_t["orderW"] = orderW
    open_max_t["quicknet"] = quicknet
    open_max_t["net"] = net
    return open_max_t