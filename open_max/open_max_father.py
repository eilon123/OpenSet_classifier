


import numpy as np
import torch


class openmax():
    num_clusters = 2

    def __init__(self,args,open_max_t):
        self.openset = args.openset
        self.notSubclass =0 and args.overclass

        self.Nclasses = int(((self.openset==1) *6 + (self.openset==0) *10)*(1+args.overclass) /(1+self.notSubclass*args.overclass))
        self.vectorWise = open_max_t["vectorWise"]
        self.net = open_max_t["net"]
        if self.openset:
            self.root = "open_max/data"
        else:
            self.root = 'open_max/fulldata'

        self.overclass = args.overclass

        self.kl = args.kl
        self.unrconst = 1
        self.feat = args.f * self.vectorWise

        self.deepclassifier = args.deepclassifier
        if self.deepclassifier:
            self.quicknet = open_max_t["quicknet"]
        # kmeans
        self.useKmeans = open_max_t["iskmeans"] and self.vectorWise and self.feat


        # pca
        self.pcaperSet = False
        self.pca = open_max_t["pca"] and self.vectorWise and self.feat and self.pcaperSet==0

        self.ownPCA = 0
        if self.ownPCA:
            self.pca = False
        self.num_comp = 8
        self.alpha = open_max_t["alpha"]
        self.layerType = "scores"
        self.layernum = -1
        self.weibull_address = 'open_max/weibull/'
        self.L = 0 and self.feat

        self.layers = ["scores","featlayer","L-2","L-3","L-4","L-5"]
        if (self.deepclassifier or self.kl) and not self.feat:
            self.layers = self.layers[:-2]
        if self.deepclassifier ==0 and self.feat==0 and self.kl ==0:
            self.layers = self.layers[0:1]
        if self.L and self.vectorWise:
            self.layers = self.layers[0:2]

    def setAddress(self, idx=-1):
        self.layernum = idx
        if idx == -1:
            self.layerType = "scores"
        elif idx == 0:

            self.layerType = "featlayer"
        elif idx == 1:
            self.layerType = "L-2"
        elif idx == 2:
            self.layerType = "L-3"
        elif idx == 3:
            self.layerType = "L-4"
        elif idx == 4:

            self.layerType = "L-5"
        print(" ")
        # if idx == -1:
        #     self.useKmeans = False
        # else:
        #     self.useKmeans = True
        print("layer is " , self.layerType)
    def setnumClusters(self,num = 2):
        self.num_clusters = num
    def calcPCA(self,X):
        X = X.numpy()
        X_meaned = X - np.mean(X, axis=0)
        cov_mat = np.cov(X_meaned, rowvar=False)

        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
        return eigen_values,eigen_vectors

    def calcdistPCA(self,eigen_values,eigen_vectors,dist ,direct):
        # sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1][:self.num_comp]
        for i in range(len(eigen_vectors)):
            if i not in sorted_index and direct:
                eigen_vectors[i] = 0
            if i in sorted_index and not direct:
                eigen_vectors[i] = 0

        return np.dot(eigen_vectors.transpose(),dist.transpose()).transpose()


    def ChooseFeat(self,img_feat,cls_indx):
        if self.vectorWise and (self.feat == 0 or self.layerType == "scores"):
            return img_feat[:self.Nclasses]
        elif self.vectorWise and (self.feat or self.layerType == 'scores'):
            return img_feat
        else:
            return img_feat[cls_indx]