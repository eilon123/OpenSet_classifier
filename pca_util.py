

from sklearn.decomposition import PCA
import wandb
from utils import progress_bar
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
class pca_util():
    def __init__(self,args ,device,net,trainloader,testloader):
        super(pca_util, self).__init__()
        self.net = net
        self.overclass = args.overclass
        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader
        wandb.watch(self.net)
    def eval(self):
        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, pool, featureLayer, _ = self.net(inputs)
                featureLayer = featureLayer.cpu().numpy()


                if batch_idx == 0:
                    features = featureLayer
                else:
                    features = np.concatenate((features, featureLayer))

                batch_idx, len(self.trainloader)

        pcaTrain = (PCA(n_components=512))
        pcaTrain.fit(features)
        eigenvalues = pcaTrain.explained_variance_
        i=0
        while True:

            if np.sum(eigenvalues[0:i+1]) > 0.9 *np.sum(eigenvalues):
                print("train 90% eiganvalue is: ",i)

                break
            else:
                i +=1

        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, pool, featureLayer, _ = self.net(inputs)
                featureLayer = featureLayer.cpu().numpy()


                if batch_idx == 0:
                    features = featureLayer
                else:
                    features = np.concatenate((features, featureLayer))

                batch_idx, len(self.trainloader)

        pcaTrain = (PCA(n_components=512))
        pcaTrain.fit(features)
        eigenvalues = pcaTrain.explained_variance_
        i=0
        while True:
            if np.sum(eigenvalues[0:i+1]) > 0.9 *np.sum(eigenvalues):
                print("test 90% eiganvalue is: ",i)
                break
            else:
                i += 1
    def evalLambda(self):
        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, pool, featureLayer, _ = self.net(inputs)
                featureLayer = featureLayer.cpu().numpy()


                if batch_idx == 0:
                    features = featureLayer
                else:
                    features = np.concatenate((features, featureLayer))

                batch_idx, len(self.trainloader)

        pcaTrain = (PCA(n_components=512))
        pcaTrain.fit(features)
        eigenvalues = pcaTrain.explained_variance_

        if not(self.overclass):
            with open('regPCA.pkl', 'wb') as f:
                pkl.dump(pcaTrain.components_, f)
                return
        else:
            with open('regPCA.pkl', 'rb') as f:
                comp = pkl.load(f)
        u0 = np.mean(features,axis=0)

        # C = [ [[] for l in range(50000)] for i in range(512)]
        C = np.dot(features-u0,(pcaTrain.components_))
        # for i, feat in enumerate(features):
        #     for k in range(512):
        #         C[k][i] = np.dot(feat-u0,comp[k])
        # lamda = list()
        # for k in range(512):
        #     lamda.append(0)
        #     for i ,_ in enumerate(features):
        #         lamda[k] += C[k][i] / 50000
        # print(lamda)
        #
        # for i, feat in enumerate(features):
        #     for k in range(512):
        #         C[k][i] = np.dot(feat - u0, pcaTrain.explained_variance_[k]*np.transpose(pcaTrain.components_[k]))
        lamda = list()
        for k in range(512):
            lamda.append(0)

            for i in range(np.shape(features)[1]):
                lamda[k] += C[k][i] ** 2 / 50000
        print(lamda)
        return lamda