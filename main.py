'''Train CIFAR10 with PyTorch.'''
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import copy
import time

import torchvision
import os
from opts import *
from Inference import *
from createDataset import *
from specials import *
from entropyloss import *
from ent2 import *

from stats import *
from models import *
import numpy as np
import wandb
from open_max import get_model_features
from open_max import compute_distances
from open_max import open_max_stats
from open_max import compute_openmax

from open_max import MAV_Compute
from open_max import open_max_father
from open_max.train_net import *


# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# os.environ["WANDB_MODE"] = "dryrun"

args = parse()

def main(party=1,m=0.2,net=0,optimizer=0,epochstop = 190):

    args.parts = party
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train, transform_test = prepData(args.mnist)
    if net==0:
        newTrain = 1
    else:
        newTrain = 0
    if net==0:
        net = chooseNet(args, device)

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            address = findAddress(args)
            checkpoint = torch.load(address + 'checkpoint/ckpt.pth')
            new_weights = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
            net.load_state_dict(new_weights)
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if args.union and args.trans:
        args.transunion = True
    else:
        args.transunion = False
    args.union = args.union if not (args.union and args.trans) else False

    # Loading dataset

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)



    if args.union==0 and args.part:
        party = 5
        print("partial data given by ", party)
        trainset = partialDataset(trainset, percentage=10 * party)
    if args.union:
        selectedClasses = np.arange(5)
        selectedClasses = np.arange(4)
    else:

        aList = np.arange((10))
        selectedClasses = np.array(np.sort(random.sample(list(aList), args.Nclasses)))
        selectedClasses = np.arange(6)
        # if args.resume and os.path.isfile(address + 'classes.xlsx') :
        #     tmplist = []
        #     csv_file = open(address + 'classes.xlsx', 'r')
        #     x = csv_file.readline()
        #     tmplist.append(int(x[0]))
        #     tmplist.append(int(x[2]))
        #     tmplist.append(int(x[4]))
        #     tmplist.append(int(x[6]))
        #     tmplist.append(int(x[8]) )
        #     tmplist.append(int(x[10]))
        #     selectedClasses = np.asarray(tmplist)
        #         # selectedClasses = np.asarray(list(pd.read_excel(address + 'classes.xlsx').columns))
        # else:
        #     selectedClasses = np.arange(6)
        #     selectedClasses = np.arange(2)
        #     # selectedClasses = np.array([2,3,4,5,6,7])
        #     if args.Nclasses == 5:
        #         selectedClasses = np.arange(5)
        #     if args.Nclasses == 2:
        #         selectedClasses = np.arange(2)
        if args.rand and args.resume == 0:
            selectedClasses = np.arange(6)
    print("Selected class are %d", selectedClasses)
    classidx_to_keep = np.array(selectedClasses)

    trainloader = createTrainloader(args, trainset, classidx_to_keep)
    testloader, fullTestset = createTestloader(args, testset, classidx_to_keep)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    if args.ph1:
        args.overclass = False

    ent = EntropyLoss(args, device)
    ent2 = Entropy2(args, device,int(len(selectedClasses)))
    if newTrain == 1:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.lamda)


    if args.trans :
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                               weight_decay=args.lamda)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    reduce = 0

    test_var=0
    train_var=0
    trainloss=0
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
                      optimizer, trainloader,
                      testloader, fullTestset.targets[:],
                      selectedClasses, reduce,
                       ent2,best_acc)

    if infer.trans:
        infer.setTrans()
    infer.setclasses(selectedClasses)
    # Epoch loop
    orderW=0
    if args.ae ==0 and args.overclass ==0 and args.resume:
        w = weight_lst(net)
        weightValue = torch.sort(w[-2])[0]
        # 5 8 11 14 17 20 23 26 29 32 35 38
        order = torch.sort(w[-2])[1]
        x = torch.sort(w[-2])[1]
        highest = []
        lowest = []
        wlowest = []
        whighest = []
        orderW = []
        for i in range(int((10 + args.overclass * 10) / (1 + args.openset))):
            lowest.append(x[i])
            highest.append(x[i])
            wlowest.append(weightValue[i][0:10])
            whighest.append(weightValue[i][-11:-1])
            orderW.append(order[i][:10])
        from statistics import mean

        print("highest mean",torch.mean(torch.cat(whighest,0)).item())
        print("highest var",torch.var(torch.cat(whighest,0)).item())

        print("lowest mean",torch.mean(torch.cat(wlowest,0)).item())
        print("lowest var", torch.var(torch.cat(wlowest,0)).item())
        infer.setWeights(highest,lowest,order)
    quicknet = infer.Getquicknet()
    if args.overclass and args.train:
        add = infer.address
        if len(args.save_path)>0:
            add = args.save_path
        csv_file = open(add + '/overclass params.xlsx', 'w')
        csv_writer = csv.writer(csv_file, delimiter=",")
        tmp = np.array(["Kunif","Kuniq"])
        csv_writer.writerow(tmp)
        tmp = np.array([str(args.Kunif),str(args.Kuniq)])
        csv_writer.writerow(tmp)
        csv_file.close()
    for epoch in range(start_epoch, start_epoch + 600):

        spc = specials(args, device, net, trainset, testset, classidx_to_keep, trainloader, testloader,
                       infer.address)
        spc.flow()
        if args.arcface:
            infer.setM(m)
        if args.train :
            if args.ae:
                infer.AEtrain(epoch)
            else:
                trainloss,train_var = infer.train(epoch, False)


        if args.train == 1:

            print("epoch num ",epoch)

            inspect = (epoch%10==0)  and (epoch > 10) and args.overclass == True
            # inspect = 0
            if args.ae:
                acc = infer.AEtest(epoch,False)
            else:
                if epoch == start_epoch + 199:
                    acc,test_loss,test_var= infer.test(epoch, True)
                else:
                    acc ,test_loss,test_var= infer.test(epoch,False,inspect=inspect)
            # if epoch == start_epoch + 199 and args.ae:
            #     acc ,test_loss,test_var = infer.AEtest(epoch, True)
            if args.ae:
                if acc < best_acc:
                    best_acc = acc
            else:
                if acc > best_acc:
                    best_acc = acc
        else:
            break
        print("================= best acc is",best_acc)

        scheduler.step()
    acc, test_loss, test_var = infer.test(epoch, True)

    quicknet = infer.Getquicknet()
    return acc,net,quicknet,selectedClasses,orderW


if __name__ == '__main__':


    entl = list()
    add = "entorpy no bias through layers"
    listdir = os.listdir(add)
    oneShot = 1
    for i,dir in enumerate(listdir):

    # kneigh = [2,5,10,20,30,50,70,100,130,160,200,250,350,500,2,5,10,20,30,50,70,100,130,160,200,250,350,500]
    # for i in range(14):
    #     dir = 'newYear/mu 0.001_lamda 0.001_class2-7'
    #     print(dir)
        print("======== run number ",i)

        if oneShot ==0:
            args.load_path = os.path.join(add,dir)
        # args.load_path = os.path.join("newshit",'mu 0.005')
        test_loss, net,quicknet,selectedClasses,orderW = main()
        if oneShot and args.resume ==0:
            exit()
        save_address= ''
        # save_address = " openset through epochs/" + str(i)
        if args.openset == 0 or args.Nclasses != 6:
            exit()
        if test_loss > 20 and 1 :
            vectoerWise=1
            iskmeans = 0
            pca=0
            args.aug=0
            open_max_t = bulid_openmax(vectorWise=vectoerWise,pca=pca,iskmeans=iskmeans,alpha=(i+1)*0.05,orderW=orderW,quicknet=quicknet,net=net,classes=selectedClasses)
            feat = get_model_features.get_model_features(args, open_max_t)
            mav = MAV_Compute.mavClass(args=args,open_max_t= open_max_t)
            dist = compute_distances.distcomupte(args,open_max_t )
            open_max = compute_openmax.compute_openmax(args, open_max_t, save_path="")
            stats = open_max_stats.openmax_stats(args,open_max_t , save_path="")

            mav.setnumClusters(i+2)
            open_max.setnumClusters(i + 2)
            dist.setnumClusters(i+2)
            if iskmeans:
                open_max.setTailSize()


            feat.restart()
            feat.get_model_features_main()
            mav.MAV_Compute_main()
            ent = dist.compute_distances_main()
            # stats.getDists()
            # stats.getNNDists()


            if len(args.load_path) > 0:
                csv_file = open(args.load_path + '/results.xlsx', 'w')
            else:
                address = findAddress(args)
                csv_file = open(address + '/results.xlsx', 'w')
            csv_writer = csv.writer(csv_file, delimiter=",")

            tmp = np.array([" ", "L", "L-1", "L-2", "L-3", "L-4", "L-5", "L-6"])
            csv_writer.writerow(tmp)


            open_max.setNN(0)
            open_max.setCosine(0)
            open_max.setDependent(0)
            auroc ,myauroc =open_max.runOpenmax(open_max,  save_address)
            tmp = np.array(["open max class independent",auroc[0],auroc[1],auroc[2],auroc[3], auroc[4],auroc[5],auroc[6]])
            csv_writer.writerow(tmp)

            open_max.setNN(0)
            open_max.setCosine(0)
            open_max.setDependent(1)
            auroc, myauroc = open_max.runOpenmax(open_max, save_address)
            tmp = np.array(["open max class dependent", auroc[0], auroc[1], auroc[2], auroc[3], auroc[4], auroc[5], auroc[6]])
            csv_writer.writerow(tmp)


            open_max.setNN(1)
            open_max.setCosine(0)
            open_max.setDependent(1)
            auroc, myauroc = open_max.runOpenmax(open_max, save_address,skip=1)
            tmp = np.array(["D-SCR NN class dependent", myauroc[0], myauroc[1], myauroc[2], myauroc[3], auroc[4], myauroc[5], myauroc[6]])
            csv_writer.writerow(tmp)


            open_max.setNN(1)
            open_max.setCosine(1)
            open_max.setDependent(1)
            auroc, myauroc = open_max.runOpenmax(open_max, save_address,skip=1)
            tmp = np.array(["Angle D-SCR NN class dependent", myauroc[0], myauroc[1], myauroc[2], myauroc[3], auroc[4], myauroc[5], myauroc[6]])
            csv_writer.writerow(tmp)


            tmp = np.array(["accuracy",test_loss," "," "," "," "," "," "])
            csv_writer.writerow(tmp)

            csv_file.close()
            if oneShot == 1:
                exit()

