


import numpy as np
import torch
from open_max.openmax_utils import *
from numpy import linalg as LA

import os
class openmax():
    num_clusters = 2

    def __init__(self,args,open_max_t):
        self.openset = args.openset
        self.notSubclass =0 and args.overclass

        self.Nclasses = int(((self.openset==1) *6 + (self.openset==0) *10)*(1+args.overclass) /(1+self.notSubclass*args.overclass))
        self.vectorWise = open_max_t["vectorWise"]
        self.net = open_max_t["net"]
        if self.openset:
            self.root = "open_max/fulldata"
        else:
            self.root = 'open_max/fulldata'
        self.load_path = args.load_path
        self.overclass = args.overclass
        self.deepPCA = True and args.pca
        self.kl = args.kl
        self.unrconst = 1
        self.feat = args.f * self.vectorWise
        self.kneighbours=2
        self.deepclassifier = args.deepclassifier
        self.classes = open_max_t["classes"]

        if self.deepclassifier:
            self.quicknet = open_max_t["quicknet"]
        # kmeans
        self.useKmeans = open_max_t["iskmeans"] and self.vectorWise and self.feat

        self.NNdists = 0
        self.NNpca = 0 and self.NNdists

        self.class_dependent = 0
        self.cosine = 0
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
        # self.aug = args.aug
        self.layers = ["scores","featlayer","L-2","L-3","L-4","L-5","L-6"]
        self.chosenLayer = "L-3"
        self.shortcut = True
        if (self.deepclassifier or self.kl) and not self.feat:
            self.layers = self.layers[:-2]
        if self.deepclassifier ==0 and self.feat==0 and self.kl ==0:
            self.layers = self.layers[0:1]
        if self.L and self.vectorWise:
            self.layers = self.layers[0:2]
        self.imgplane = 0
    def setDependent(self,isdep):
        self.class_dependent = isdep
    def setCosine(self,iscos):
        self.cosine = iscos
    def setNN(self,isNN):
        self.NNdists = isNN
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
        elif idx == 5:

            self.layerType = "L-6"

        print(" ")
        # if idx == -1:
        #     self.useKmeans = False
        # else:
        #     self.useKmeans = True
        print("layer is " , self.layerType)
    def setdeepNNpca(self,entry):
        self.NNpca = entry
    def setnumClusters(self,num = 2):
        self.num_clusters = num
    def setK(self,num):
        self.kneighbours = num
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


    def ChooseFeatLayer(self,img_feat,cls_indx,layerType):
        if self.vectorWise and (self.feat == 0 or layerType == "scores"):
            return img_feat[:self.Nclasses]
        elif self.vectorWise and (self.feat or layerType == 'scores'):
            return img_feat
        else:
            return img_feat[int(cls_indx)]
    def ChooseFeat(self,img_feat,cls_indx):
        if self.vectorWise and (self.feat == 0 or self.layerType == "scores"):

            return img_feat[self.classes]
        elif self.vectorWise and (self.feat or self.layerType == 'scores'):
            return img_feat
        else:
            return img_feat[int(cls_indx)]
    def Choose_address(self):
        if len(self.load_path) >0:
            address = 'open_max/saved_data/' + self.load_path + '/'
        elif self.overclass:
            address = 'open_max/saved_data/overclass/'
        elif self.useKmeans:
            address = 'open_max/saved_data/kmeans/'
        elif self.pca:
            address = 'open_max/saved_data/pca/'
        else:
            address = 'open_max/saved_data/regular/'
        address = 'open_max/'
        return address
    def runOpenmax(self,open_max,save_address,skip=0):
        aurocl = []
        myaurocl = []
        for layernum in range(-1,6):
            if skip and layernum > 2:
                aurocl.append(-1)
                myaurocl.append(-1)
                continue
            open_max.setAddress(layernum)
            myauroc, auroc = open_max.compute_openmax_main(save_address)
            print("my auroc is ", np.max([myauroc,1-myauroc]))
            print("openmax auroc is ", np.max([auroc,1-auroc]))
            aurocl.append(np.round(100*np.max([auroc,1-auroc]),2))
            myaurocl.append(np.round(100*np.max([myauroc,1-myauroc]),2))
        return aurocl,myaurocl
    # open_max.runOpenmax(open_max, -1, save_address, aurocl, myaurocl)
    #
    # open_max.runOpenmax(open_max, 0, save_address, aurocl0, myaurocl0)
    #
    # if args.deepclassifier or args.f:
    #     open_max.runOpenmax(open_max, 1, save_address, aurocl1, myaurocl1)
    #     open_max.runOpenmax(open_max, 2, save_address, aurocl2, myaurocl2)  # L-3
    #     open_max.runOpenmax(open_max, 3, save_address, aurocl3, myaurocl3)
    #     open_max.runOpenmax(open_max, 4, save_address, aurocl4, myaurocl4)
    #     open_max.runOpenmax(open_max, 5, save_address, aurocl5, myaurocl5)
    def BuildfeatMatrix(self,data_type,layer,numclasses,numParameters,shortver=0,clsnum=0):
        dataSize = 5000
        if data_type != 'train' or shortver:
            dataSize=1000
        if data_type == 'aug':
            dataSize = 10

        mat = np.zeros(shape=(numclasses*dataSize,numParameters))
        i=0
        k=0
        targets = []
        feat_dir = self.feature_dir.replace(data_type,"")
        for class_no in os.listdir(os.path.join(feat_dir, data_type, self.layers[0])):
            if clsnum !=0 and clsnum != class_no:
                continue
            featurefile_list = os.listdir(os.path.join(feat_dir, data_type, layer, class_no))

            for featurefile in featurefile_list:
                targets.append(class_no)
                Feat = torch.from_numpy(
                    np.load(os.path.join(feat_dir, data_type, layer, class_no, featurefile)))
                mat[i] = Feat
                i+=1
                k+=1
                if k ==1000 and shortver:
                    k=0
                    break

        return mat,targets
    def minDist(self,FeatMat,trainmat,numclasses,aug,numOfmins=20,Ntest=6):
        if numOfmins < 1:
            numOfmins=1
        trainexamples = 5000
        exmaples_Num = 1000
        isTrain = 0
        if aug:
            exmaples_Num = 10
        elif np.array_equal(FeatMat, trainmat):
            exmaples_Num = 1000
            isTrain = True
            trainexamples =1000
            Ntest = 1
        N = self.Nclasses
        if numclasses == 1:
            N = 1
            # Ntest=1
        else:
            N = self.Nclasses
        # tmppp for train check
        # trainexamples = 2500
        # Ntest = 1
        # exmaples_Num = 2500
        Feat2 = np.transpose(np.tile(np.linalg.norm(FeatMat,axis=1)**2,(N*trainexamples,1)))

        corr2 = np.tile(np.linalg.norm(trainmat, axis=1)**2, (Ntest * exmaples_Num, 1))
        corrTestTrain = -2*np.matmul(FeatMat,np.transpose(trainmat))
        dist = corrTestTrain + Feat2 + corr2
        dist[dist < 0] = 0
        from scipy import spatial
        cosine = []

        mindist =np.sum( np.sort(np.sqrt(dist),axis=1)[:,0:numOfmins],axis=1)
        # if isTrain:
        #     mindist = mindist[:,1:]
        # idx = np.argmin(np.sqrt(corrTestTrain + Feat2 + corr2),axis=1)
        idx = torch.sort(torch.sqrt(torch.Tensor(dist)), axis=1)[1][:, :numOfmins]
        # j=0
        # idx = torch.sort(torch.sqrt(torch.Tensor(dist)), axis=1)[1]
        # featnorm = np.linalg.norm(FeatMat,axis=1)
        # trainnorm = np.linalg.norm(trainmat, axis=1)
        if self.cosine:
            for ii,f in enumerate(FeatMat):
            #     # print(ii)
            #     # for f in trainmat:
            #     #     cosine.append(np.arccos(1-spatial.distance.cosine(f,ft))*(180/np.pi))
                if np.array_equal(FeatMat, trainmat):
                    cosine = (np.arccos(1 - spatial.distance.cosine(f, trainmat[idx[ii, 1]])) * (180 / np.pi))
                else:
                    cosine = (np.arccos(1 - spatial.distance.cosine(f, trainmat[idx[ii,0]])) * (180 / np.pi))
            #
                mindist[ii] = cosine



        if self.NNpca:
            from sklearn.decomposition import PCA
            base = PCA(n_components=np.min([len(FeatMat[j]),numOfmins]))
            for i in idx:
                base.fit(trainmat[i][:])
                mindist[j] = np.linalg.norm(base.transform(np.reshape(FeatMat[j], (1, len(FeatMat[j]))))) -np.linalg.norm(FeatMat[j])
                j+=1
                if j == 6000:
                    dbg=3
        # layer = 'L-4'
        # mindistl = []
        # for  idx in range(len(mindist)):
        #     Feat = FeatMat[idx]
        #     First = True
        #     for class_no in os.listdir(os.path.join(self.feature_dir, 'train', self.layers[0])):
        #         featurefile_list = os.listdir(os.path.join(self.feature_dir, 'train', layer, class_no))
        #         for featurefile in featurefile_list:
        #             Neigh = torch.from_numpy(
        #                 np.load(os.path.join(self.feature_dir, 'train', layer, class_no, featurefile)))
        #
        #             dist = compute_distance(self.ChooseFeatLayer(Feat, int(class_no), layerType=layer),
        #                                     self.ChooseFeatLayer(Neigh, int(class_no), layerType=layer),
        #                                     distance_type=self.distance_type)
        #             if First:
        #                 mindist2 = dist
        #                 First = False
        #             if dist < mindist2:
        #                 mindist2 = dist
        #
        #
        #     mindistl.append(mindist2)
        return mindist,idx

    def pcadist(self,c,img_features,i,predictor):

        if self.pca and self.deepPCA:
            distance = 0
            for layernum in range(1, 4):
                dist = c[layernum-1] - img_features[layernum-1]
                if self.pca:
                    distpca = predictor[layernum - 1][i].transform(np.reshape(dist, (1, len(dist)))).flatten()
                    distpca = LA.linalg.norm(distpca)
                    distance += distpca
        elif self.pca:
            dist = c - img_features
            distpca = predictor.transform(np.reshape(dist, (1, len(dist))))
            dist_pca_partial = np.pad(distpca[0][0:self.num_comp], int(len(dist) - self.num_comp), 'constant',
                                      constant_values=(None, 0))
            nan_array = np.isnan(dist_pca_partial)
            not_nan_array = ~ nan_array
            dist_pca_partial = dist_pca_partial[not_nan_array]
            distance = LA.norm(dist_pca_partial) * self.alpha + (1 - self.alpha) * np.sqrt(
                np.abs(LA.norm(dist) - LA.norm(distpca)))
            # distance = LA.norm(dist_orth * self.alpha + (1 - self.alpha) * (distpca))

        return distance


