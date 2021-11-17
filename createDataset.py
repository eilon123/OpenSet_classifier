import numpy as np
import torch
import copy
from models import *
import os
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data

def changeTar(set,dict):
    for i in range(len(set.targets)):
        set.targets[i] = dict[int(set.targets[i])]
    tmp = copy.deepcopy(set.classes)
    for i in range(len(set.classes)):
        set.classes[i] = tmp[dict[i]]
    return set
def createDataset(trainset, classidx_to_keep, isTrain,batchSize,imbalance=False, test=False):
    targets = trainset.targets[:]
    outputSet = copy.deepcopy(trainset)
    ######################## first network data
    idx_to_keep = []
    if test:
        for i in range(len(targets)):
            if targets[i] not in classidx_to_keep:
                outputSet.targets[i] = 9
        outputSet.data = outputSet.data[i]
    else:
        for classKeep in classidx_to_keep:
            idxClass = list(np.where(targets == classKeep))
            idxClass = idxClass[0].tolist()
            if imbalance and classKeep%2:
                del idxClass[int(0.5*len(idxClass)-1):-1]
            idx_to_keep += idxClass
        idx_to_keep.sort()

        outputSet.targets = [outputSet.targets[i] for i in idx_to_keep]
        outputSet.data = [outputSet.data[i] for i in idx_to_keep]
    trainloader = torch.utils.data.DataLoader(
        outputSet, batch_size=batchSize, shuffle=isTrain, num_workers=8)
    return trainloader,outputSet


def partialDataset(trainset, percentage):
    indices = np.arange(len(trainset))
    train_indices, _ = train_test_split(indices, train_size=percentage, stratify=trainset.targets)

    # Warp into Subsets and DataLoaders
    train_dataset = torch.utils.data.Subset(trainset, train_indices)
    return train_dataset


def createTrainloader(args, trainset, classidx_to_keep=0):
    if args.openset:
        trainloader,_ = createDataset(trainset, classidx_to_keep, True,args.batch)
    elif args.trans:
        trainloader2nd = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch, shuffle=False, num_workers=8)
        fulltrainSet = copy.deepcopy(trainset)
        trainloader,_ = createDataset(fulltrainSet, classidx_to_keep, True,args.batch)
        # trainloader2nd = trainloader
        return trainloader, trainloader2nd
    else:
        if args.union:
            if args.imbalance:

                _,trainset = createDataset(trainset, np.array([0,1,2,3,4,5,6,7,8,9]), isTrain=True,batchSize=1,imbalance=args.imbalance, test=False)


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
    return trainloader, 0


def createTestloader(args, testset,classidx_to_keep=0):
    fullTestset = copy.deepcopy(testset)
    if args.oTest:

        testloader,_ = createDataset(testset, classidx_to_keep, False, args.batch)
        fullTestset = testset
    elif args.trans and not(args.directTrans):

        testloader,_ = createDataset(testset, classidx_to_keep, True,args.batch)

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
    d = args.mnist + 3*(args.mnist ==0)
    if args.extraLayer:
        print("extraLayer")
        net = ResNet18(num_classes=5, extraLayer=2, extraClasses=1 + (
                extraClasses - 1) * args.overclass,d=d)  # extraClasses=1 + (extraClasses - 1) * args.overclass)
    elif args.union and args.overclass:
        net = ResNet18(num_classes=10,d=d)  # extraClasses=1 + (extraClasses - 1) * args.overclass)
    elif args.union:
        net = ResNet18(num_classes=10,d=d)
    elif args.overclass and args.extraLayer:
        net = ResNet18(num_classes=10, extraClasses=1 + (extraClasses - 1) * args.overclass,d=d)
    elif args.overclass and args.featperclass and not args.trans:
        net = ResNet18(num_classes=10, extraClasses=1 + (extraClasses - 1) * args.overclass, extraFeat=True,pool = args.pool,d=d)
    else:
        net = ResNet18(num_classes=10 + 10 * (args.extraclass - 1) * args.overclass  * (args.trans==0)*(args.openset ==0),pool = args.pool,d=d)
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
    elif args.ph2 or args.ph1:
        print("loading ph1 for ph2")
        address += 'ph1/checkpoint/ckpt.pth'
    elif args.extraclass >2:
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
        if not os.path.isdir('extraLayer'):
            os.mkdir('extraLayer', )
        if not os.path.isdir('extraLayer/checkpoint'):
            os.mkdir('extraLayer/checkpoint')
        address += 'extraLayer/checkpoint/ckpt.pth'
    if args.ph1:
        if not os.path.isdir('ph1'):
            os.mkdir('ph1', )
        if not os.path.isdir('ph1/checkpoint'):
            os.mkdir('ph1/checkpoint')
        address += 'ph1/checkpoint/ckpt.pth'

    elif args.ph2:
        if not os.path.isdir('ph2'):
            os.mkdir('ph2', )
        if not os.path.isdir('ph2/checkpoint'):
            os.mkdir('ph2/checkpoint')
        address += 'ph2/checkpoint/ckpt.pth'
    elif args.extraclass >2:
        if not os.path.isdir('5overclass'):
            os.mkdir('5overclass', )
        if not os.path.isdir('5overclass/checkpoint'):
            os.mkdir('5overclass/checkpoint')
        address += '5overclass/checkpoint/ckpt.pth'
    elif args.trans and args.overclass and args.featperclass:
        if not os.path.isdir('featpertrans'):
            os.mkdir('featpertrans')
        if not os.path.isdir('featpertrans/checkpoint'):
            os.mkdir('featpertrans/checkpoint')
        address += 'featpertrans/checkpoint/ckpt.pth'
    elif args.trans and args.overclass:
        if not os.path.isdir('transover'):
            os.mkdir('transover')
        if not os.path.isdir('transover/checkpoint'):
            os.mkdir('transover/checkpoint')
        address += 'transover/checkpoint/ckpt.pth'
    elif args.trans:
        if not os.path.isdir('transfer'):
            os.mkdir('transfer' )
        if not os.path.isdir('transfer/checkpoint'):
            os.mkdir('transfer/checkpoint')
        address += 'transfer/checkpoint/ckpt.pth'
    elif args.imbalance:
        if not os.path.isdir('imbalance'):
            os.mkdir('imbalance' )
        if not os.path.isdir('imbalance/checkpoint'):
            os.mkdir('imbalance/checkpoint')
        address += 'imbalance/checkpoint/ckpt.pth'
    elif args.union and args.overclass:
        if not os.path.isdir('unionOC'):
            os.mkdir('unionOC' )
        if not os.path.isdir('unionOC/checkpoint'):
            os.mkdir('unionOC/checkpoint')
        address += 'unionOC/checkpoint/ckpt.pth'
    elif args.union and not args.overclass:
        if not os.path.isdir('union'):
            os.mkdir('union')
        if not os.path.isdir('union/checkpoint'):
            os.mkdir('union/checkpoint')
        address += 'union/checkpoint/ckpt.pth'
    elif args.overclass and args.featperclass:
        if not os.path.isdir('featperclass'):
            os.mkdir('featperclass')
        if not os.path.isdir('featperclass/checkpoint'):
            os.mkdir('featperclass/checkpoint')
        address += 'featperclass/checkpoint/ckpt.pth'
    elif args.openset:
        if not os.path.isdir('openset'):
            os.mkdir('openset')
        if not os.path.isdir('openset/checkpoint'):
            os.mkdir('openset/checkpoint')
        address += 'openset/checkpoint/ckpt.pth'
    elif args.overclass:
        if not os.path.isdir('overclass'):
            os.mkdir('overclass')
        if not os.path.isdir('overclass/checkpoint'):
            os.mkdir('overclass/checkpoint')
        address += 'overclass/checkpoint/ckpt.pth'

    else:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        address += 'checkpoint/ckpt.pth'
    return address

def weight_lst(net):
    """
        :param self.
        :return: A list of iterators of the network parameters.
    """
    return [w for w in net.parameters()]