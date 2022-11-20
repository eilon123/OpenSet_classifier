'''Train CIFAR10 with PyTorch.'''
import matplotlib.pyplot as plt
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
from ent2 import *

from stats import *
from models import *
import numpy as np
from scipy.special import softmax
import wandb
import random
from open_max import get_model_features
from open_max import compute_distances
from open_max import open_max_stats
from open_max import compute_openmax

from open_max import MAV_Compute
from open_max import open_max_father



# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# os.environ["WANDB_MODE"] = "dryrun"

args = parse()
def openmaxroutine(epoch,acc):
    if epoch % 10 == 0 and acc > 30:
        save_address = ""
        feat.restart()
        feat.get_model_features_main()
        mav.MAV_Compute_main()
        dist.compute_distances_main()
        open_max.setAddress(0)
        myauroc, auroc = open_max.compute_openmax_main(save_address)
        aurocl0.append(auroc)
        myaurocl0.append(myauroc)
        open_max.setAddress(-1)
        myauroc, auroc = open_max.compute_openmax_main(save_address)
        aurocl.append(auroc)
        myaurocl.append(myauroc)
        res.append(acc)
        print(aurocl0)
        print(myaurocl0)
        print(aurocl)
        print(myaurocl0)
        print(res)
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

    args.union = args.union if not (args.union and args.trans) else False

    # Loading dataset

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)


    if args.rand:
        # dict = np.random.randint(10, size=10)
        dict = np.array([4, 3, 7, 2, 6, 1, 0, 8, 5, 9])
        trainset = changeTar(trainset, dict)
        testset = changeTar(testset, dict)
    party = 1
    if args.part and party < 10:
        trainset = partialDataset(trainset, percentage=5000*party)
        # testset = partialDataset(testset)
    if args.union and args.overclass and args.openset:
        selectedClasses = np.arange(6)
    elif args.union:
        selectedClasses = np.arange(5)
        # selectedClasses = np.arange(2)
    else:
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
    ent2 = Entropy2(args, device)
    # args.lamda = 0
    if newTrain == 1:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.lamda)


    if args.trans :
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                               weight_decay=5e-4)

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
                      classes, reduce,
                       ent2,best_acc)

    if infer.trans:
        infer.setTrans()
    infer.setclasses(selectedClasses)
    # Epoch loop
    orderW=0
    if args.ae ==0 and args.overclass ==0:
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
    auroc0 = list()
    myauroc0 = list()
    aurocl= list()
    myaurocl = list()
    res = list()
    vectoerWise = 1
    iskmeans = 0
    pca = 1
    open_max_t = bulid_openmax(vectorWise=1, pca=0, iskmeans=0, alpha=(0 + 1) * 0.05, orderW=orderW,
                               quicknet=quicknet, net=net)
    feat = get_model_features.get_model_features(args, open_max_t)
    mav = MAV_Compute.mavClass(args=args, open_max_t=open_max_t)
    dist = compute_distances.distcomupte(args, open_max_t)
    open_max = compute_openmax.compute_openmax(args, open_max_t, save_path="")
    for epoch in range(start_epoch, start_epoch + 1000):

        spc = specials(args, device, net, trainset, testset, classidx_to_keep, trainloader, testloader,
                       infer.address)
        # spc.flow()
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

            if args.ae:
                acc = infer.AEtest(epoch,False)
            else:
                if epoch == start_epoch + 199:
                    acc,test_loss,test_var= infer.test(epoch, True)
                else:
                    acc ,test_loss,test_var= infer.test(epoch,False,inspect=inspect)
            if epoch == start_epoch + 199 and args.ae:
                acc ,test_loss,test_var = infer.AEtest(epoch, True)
            if args.ae:
                if acc < best_acc:
                    best_acc = acc
            else:
                if acc > best_acc:
                    best_acc = acc
        else:
            break


        if epoch == start_epoch + 1000:
            acc ,test_loss,test_var = infer.AEtest(epoch, True)


        print("================= best acc is",best_acc)



        scheduler.step()
    acc, test_loss, test_var = infer.test(epoch, True)
    quicknet = infer.Getquicknet()
    return trainloss,acc,train_var,test_var,net,quicknet,orderW


if __name__ == '__main__':

    aurocl = list()
    aurocl0 = list()
    aurocl1 = list()
    aurocl2 = list()
    aurocl3 = list()
    aurocl4 = list()

    myaurocl0 = list()
    myaurocl1 = list()
    myaurocl2 = list()
    myaurocl3 = list()
    myaurocl4 = list()

    # lisdir = os.listdir("arcfaceiter")
    # for i,dir in enumerate(lisdir):
    for i in range(20):
        print("======== run number ",i)
        # args.load_path = os.path.join("arcfaceiter",dir)
        # trainloss,test_loss,train_var,test_var ,net,quicknet= main(m = 0.005*(i+1))
        # trainloss, test_loss, train_var, test_var, net ,quicknet= main(m=0.005 * (15 + 1))
        if i == 0:
            trainloss, test_loss, train_var, test_var, net, quicknet,orderW = main()
        exit()
    #     save_address= ''
    #     # save_address = " openset through epochs/" + str(i)
    #     if test_loss > 20 :
    #         vectoerWise=1
    #         iskmeans = 1
    #         pca=0
    #         open_max_t = bulid_openmax(vectorWise=vectoerWise,pca=pca,iskmeans=iskmeans,alpha=(i+1)*0.05,orderW=orderW,quicknet=quicknet,net=net)
    #         feat = get_model_features.get_model_features(args, open_max_t)
    #         mav = MAV_Compute.mavClass(args=args,open_max_t= open_max_t)
    #         dist = compute_distances.distcomupte(args,open_max_t )
    #         open_max = compute_openmax.compute_openmax(args, open_max_t, save_path="")
    #         stats = open_max_stats.openmax_stats(args,open_max_t , save_path="")
    #         # stats.getoneCDF()
    #
    #
    #         mav.setnumClusters(i+2)
    #         open_max.setnumClusters(i + 2)
    #         dist.setnumClusters(i+2)
    #         if iskmeans:
    #             open_max.setTailSize()
    #         # if i ==0:
    #         #     feat.restart()
    #         #     feat.get_model_features_main()
    #         # # stats.getTSNE()
    #         #
    #         # else:
    #         #     feat.restart(0)
    #         feat.restart(0)
    #         mav.MAV_Compute_main()
    #         dist.compute_distances_main()
    #
    #         if pca==0 and iskmeans ==0:
    #             open_max.setAddress(-1)
    #             myauroc,auroc = open_max.compute_openmax_main(save_address)
    #             print(auroc)
    #             aurocl.append(auroc)
    #         # stats.routine()
    #         if vectoerWise:
    #             open_max.setAddress(0)
    #             myauroc,auroc = open_max.compute_openmax_main(save_address)
    #             aurocl0.append(auroc)
    #             myaurocl0.append(myauroc)
    #             print(aurocl0)
    #             stats.setAddress(0)
    #
    #
    #
    #         if args.deepclassifier or args.f:
    #             open_max.setAddress(1)
    #             myauroc,auroc = open_max.compute_openmax_main()
    #             print(auroc)
    #             aurocl1.append(auroc)
    #             myaurocl1.append(myauroc)
    #
    #             open_max.setAddress(2)
    #             myauroc,auroc = open_max.compute_openmax_main()
    #             print(auroc)
    #             aurocl2.append(auroc)
    #             myaurocl2.append(myauroc)
    #
    #             open_max.setAddress(3)
    #             myauroc,auroc = open_max.compute_openmax_main()
    #             print(auroc)
    #             aurocl3.append(auroc)
    #             myaurocl3.append(myauroc)
    #
    #             open_max.setAddress(4)
    #             myauroc,auroc = open_max.compute_openmax_main()
    #             print(auroc)
    #             aurocl4.append(auroc)
    #             myaurocl4.append(myauroc)
    #
    #         # closedsetlst.append(prec[0])
    #         #
    #         # misstakenlst.append(prec[4])
    #         # opensetLst.append(opensetprec)
    #     else:
    #         x=-1
    #
    #
    # print(aurocl)
    # print("featlayer")
    # print(aurocl0)
    # print("L-2")
    # print(aurocl1)
    # print("L-3")
    # print(aurocl2)
    # print("L-4")
    # print(aurocl3)
    # print("L-5")
    # print(aurocl4)
    #
    #
    # print("my foking mehtod")
    # print("featlayer")
    # print(myaurocl0)
    # print("L-2")
    # print(myaurocl1)
    # print("L-3")
    # print(myaurocl2)
    # print("L-4")
    # print(myaurocl3)
    # print("L-5")
    # print(myaurocl4)
    #
    # # print(closedsetlst)
    # # print(misstakenlst)
    # # print(opensetLst)
    # # plt.plot(aurocl)
    # # plt.savefig("auroc")
    # # plt.plot(closedsetlst)
    # # plt.savefig("closed")
    # #
    # # plt.plot(misstakenlst)
    # # plt.savefig("misstaken")
    # #
    # # plt.plot(opensetLst)
    # # plt.savefig("openset")
    # # print("train loss " ,trainlst)
    #
    # print("test loss ", testlst)
    #
    # print("train var ", trainvarlst)
    #
    # print("test var ", testvarlst)
