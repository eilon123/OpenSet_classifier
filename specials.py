from nearestNeighbour import *
from pca_util import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from Osvm import *


class specials:
    def __init__(self, args, device, net, trainset, testset, classidx_to_keep, trainloader, testloader, address):
        self.NN = args.NN
        self.osvm = args.osvm
        self.deepNN = args.deepNN
        self.pca = args.pca
        self.level = args.level
        self.orth = args.orth
        self.trans = args.trans
        self.device = device
        self.net = net

        self.trainset = trainset
        self.classidx_to_keep = classidx_to_keep
        self.testset = testset
        self.trainloader = trainloader
        self.testloader = testloader
        self.overclass = args.overclass

        self.pca_t = pca_util(args, self.device, self.net, self.trainloader, self.testloader)
        self.cls = osvm(args, self.device, self.net, self.trainset, self.testloader, self.testset.targets[:], address)

    def flow(self):
        x=self.deepNN_func()
        self.osvm
        self.pca_util
        self.level
        self.orth
    def deepNN_func(self):
        if not (self.NN):
            return
        classes = np.arange(10)
        centroids = getCentroids(self.net, self.device, self.trainset, self.classidx_to_keep)
        newTestset = copy.deepcopy(self.trainset)
        newTestset.targets = torch.utils.data.ConcatDataset([self.trainset.targets, self.testset.targets])
        newTestset.data = torch.utils.data.ConcatDataset([self.trainset.data, self.testset.data])

        # tsne, TSNEcenteroids = getCentroidsTSNE(self.net, self.device, newTestset, self.classidx_to_keep)
        # correctCent = classify(self.net, self.device, self.testloader, centroids)
        # correcttsne = classifyTSNE(tsne, self.testloader, TSNEcenteroids, len(self.trainset), self.device)
        # print("centriod correct is: ", correctCent)
        # print("tsne correct is: ", correcttsne)

        centroidsFeat,_, centroidsL1, centroidsL2, centroidsL3, centroidsL4 = getCentroids(self.net, self.device,
                                                                                         self.trainset,
                                                                                         classes)
        classify(self.net, self.device, self.testloader, centroidsFeat, centroidsL1, centroidsL2, centroidsL3,
                 centroidsL4)

    def osvm(self):
        if not (self.osvm):
            return
        # overfit
        # cls = osvm(args, device, net, trainset, trainloader1, trainset.targets[:],inf.address)
        set = self.trainset
        if self.overclass:
            set = self.cls.createoverClassDataSet()

        # testloader,testset = createDataset( testset,np.array([0,1,2]),isTrain=False, batchSize=args.batch,
        #                                test=False)
        # cls.setTestloader(testloader,testset.targets)
        clf = self.cls.getSupport(set)
        self.cls.testSVM(clf)

    def pca_util(self):
        if not (self.pca):
            return

        self.pca_t.eval()
        self.pca_t.evalLambda()

    def level(self):
        if not (self.level):
            return

        set = self.cls.createoverClassDataSet()
        if self.overclass and self.trans or not (self.overclass):
            classidx_to_keep = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        elif self.overclass:
            classidx_to_keep = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        centroids, clusterVar = getCentroids(self.net, self.device, set, classidx_to_keep)
        disth = np.zeros(shape=(20, 20))
        x = np.sqrt(clusterVar)
        var = np.sum(x, axis=1)
        print("var is: ", var)
        for row, cent in enumerate(centroids):
            if self.overclass and row % 2:
                continue
            for col, _ in enumerate(centroids):
                disth[row, col] = (LA.norm(cent - centroids[col], ord=2))
                # tmp = np.delete(disth[row,:],row)
            if self.overclass:
                print("dist to sub class is ", disth[row, row + 1])
                tmp = np.delete(disth[row, :], row + 1)

            tmp = disth[disth != 0]

            print("avg dist to classes is", np.average(tmp))
        print(disth)

    def orth(self):
        if not (self.orth):
            return
        centroids, _, _, _, _ = getCentroids(self.net, self.device, self.trainset, self.classidx_to_keep)
        hist = gethistproduct(self.net, self.device, self.trainloader, centroids)
