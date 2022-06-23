import csv
import  torch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sklearn.cluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
def weight_lst(self):
    """
        :param self.
        :return: A list of iterators of the network parameters.
    """
    return [w for w in self.parameters()]

def calcStats(featurelist,address,net):
      var_mean = torch.var_mean(featurelist, unbiased=False, dim=0)
      var = var_mean[0].cpu().numpy()
      mean = var_mean[1].cpu().numpy()
      var_df = pd.DataFrame(var)
      mean_df = pd.DataFrame(mean)
      var_df.to_csv(address + 'var.csv')
      mean_df.to_csv(address + 'mean.csv')
      Weight_lst = weight_lst(net)
      lastweight = Weight_lst[-2].cpu().detach().numpy()
      lwdf = pd.DataFrame(lastweight)
      lwdf.to_csv(address + 'weights.csv')

def calcHist(outputs,predicted,histIdx,gradeHist,extraClass):
    for i in range(len(outputs)):
        idx = int(histIdx[predicted[i].item()])
        pred = predicted[i].item()
        predProb = F.softmax(outputs[i][extraClass * pred:extraClass * pred + extraClass], dim=0)
        gradeHist[idx][extraClass * pred:extraClass * pred + extraClass] = predProb
        histIdx[predicted[i].item()] += 1
    return gradeHist
def showHist(gradeHist, address ,numHist,union,epoch=''):
    fig, axs = plt.subplots(2, int(numHist/2))
    k = 0
    j = 0
    for i in range(numHist):

        gradeClass = gradeHist[:, i].cpu().numpy()
        gradeClass = gradeClass.flatten()
        gradeClass = gradeClass[gradeClass != -1]
        hist = np.histogram(gradeClass,bins=20,range=(0,1))
        axs[k, j].hist(gradeClass, bins=20,range=(0,1))

        strg = 'class ' + str(int(i / 2) ) + "  "
        axs[k, j].set_title(strg)

        k += 1

        if k > 1:
            k = 0
            j += 1

        if union and i == 8:
            break
    plt.savefig(address + 'histogram ' + str(epoch))



def get_score_table(scoreslist,all_tar,address,all_preds=0,misstake=0,partial=1):

    if partial==0:
        addressMean = address[:-8] + "scores_table.csv"
        addressVar = address[:-8] +"var_tables.csv"
        csv_file = open(addressMean, 'w')
        var_file = open(addressVar, 'w')

        csv_writer = csv.writer(csv_file, delimiter=",")
        var_writer = csv.writer(var_file, delimiter=",")


    for cls in range(10):
        cnt=0
        if partial and cls:
            break
        scoreTot = scoreslist[0] * 0
        scoreVar = scoreslist[0] * 0

        for i,score in enumerate(scoreslist):
            if all_tar[i] != cls:
                continue
            if misstake :
                if all_tar[i].item() != all_preds[i]:
                    continue
            cnt += 1
            scoreTot += score


            # arr.append(std)
        meanscore = scoreTot /cnt

        for i,score in enumerate(scoreslist):
            if all_tar[i] != cls:
                continue
            scoreVar += (score - meanscore)**2
        scoreVar = scoreVar /cnt
        scoreVar = np.sqrt(scoreVar.cpu().detach().numpy().ravel())
        scoreVar = np.round(scoreVar,2)
        meanscore = meanscore.cpu().detach().numpy().ravel()
        meanscore = np.round(meanscore,2)
        if partial==0:
            csv_writer.writerow(meanscore)
            var_writer.writerow(scoreVar)

    if partial ==0:
        csv_file.close()
        var_file.close()
    else:
        return meanscore,scoreVar
def get_sumScoreStats(scoreslist,all_tar,address):

    addressMean = address[:-8] + "sum stats.csv"

    csv_file = open(addressMean, 'w')


    csv_writer = csv.writer(csv_file, delimiter=",")



    for cls in range(10):
        cnt=0
        scoreTot = list()
        for i,score in enumerate(scoreslist):
            if all_tar[i] != cls:
                continue
            cnt += 1
            scoreTot .append( torch.sum(score).item())


            # arr.append(std)
        meansum = np.round(np.mean(scoreTot),6)
        varsum = np.round(np.sqrt(np.var(scoreTot)),6)
        maxSum = np.round(np.max(scoreTot),6)
        minSum = np.round(np.min(scoreTot),6)

        csv_writer.writerow(np.array([meansum,varsum,maxSum,minSum]))



    csv_file.close()



def get_L2_loss(net):
    l2_reg = None
    for W in net.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def calcFeatCorr(featurelist,order,targets,deepness=0):
    corrArray = torch.zeros(featurelist.shape[1]).cuda()

    N = len(featurelist)
    targets = targets[:N]

    print(featurelist.shape[1])
    for i in range(10):
        target_i = (targets == i) * 1
        y_bar = (target_i).sum() / N
        y_bar_array = y_bar * torch.ones(N).cuda()
        y_std =torch.sqrt(((1-y_bar) ** 2 * len(targets == i) + y_bar ** 2 * len(targets != i)) / N)
        corr_ind = 0
        for feat in range(featurelist.shape[1]):
        # for j,feat in enumerate(order[i]):
            feattmp =torch.transpose(featurelist,0,1)[feat]
            feat_bar = torch.mean(feattmp)*torch.ones(N).cuda()

            feat_std =  torch.sqrt(torch.var(feattmp))

            # for example in range(featurelist.shape[0]):
            #     corrArray[corr_ind] += (featurelist[example][feat]-feat_bar[0])*(int(targets[example]==i) - y_bar)

            corrArray[corr_ind] = torch.sum((feattmp - feat_bar) * ((target_i) - y_bar_array))
            corrArray[corr_ind] /= (N *(feat_std * y_std))
            # corrArray = np.correlate(feattmp.cpu().numpy(),target_i.cpu().numpy())

            corr_ind+=1

        print("saving")
        x = corrArray.sort()[0]
        plt.plot(x.cpu())
        s = "deepness is " + str(deepness)
        plt.savefig(s)
        break


def calcinFeatCorr(featurelist,targets,order):
    # corrArray = torch.zeros(size=(featurelist.shape[1],featurelist.shape[1])).cuda()
    perClass = False
    flen = featurelist.shape[1]
    if perClass:
        newFeatList = torch.transpose(featurelist[targets==0],0,1)
    else:
        newFeatList = torch.transpose(featurelist,0,1)
    num_clusters=10
    corr_sorted = torch.zeros(size=(flen,flen))
    for i in range(10):
        if perClass:
            featperclass = featurelist[targets==i]
        else:
            featperclass = featurelist
        orderperclass = order[i]
        N = len(featperclass)
        feat_bar = torch.mean(featperclass,dim=0)
        feat_bar = feat_bar.repeat(N, 1)
        feat_std = torch.sqrt(torch.var(featperclass,dim=0))
        feat_std = feat_std.repeat(flen, 1)
        feat_std = feat_std @ torch.transpose(feat_std,0,1)
        # corrArray = ((torch.transpose(featperclass - feat_bar,0,1) @ (featperclass - feat_bar)) / feat_std)
        # corrArray/= N

        corrArray = np.abs(np.corrcoef(torch.transpose(featperclass,0,1).cpu().numpy()))
        corrArray = torch.tensor(corrArray)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(corrArray)
        # kmeans = sklearn.cluster.SpectralClustering(n_clusters=num_clusters, random_state=0).fit(corrArray)
        ii=0
        x= list()

        # for jj in  range(flen):
        #     z = (corrArray[0] - corrArray[jj]).sum()
        #
        #     if kmeans.labels_[jj] == 7:
        #         print(z)
        for j in range(num_clusters):
            cnt = 0
            for jj in range(flen):
                if kmeans.labels_[jj] == j:
                    x.append(jj)
                    cnt += 1
            plt.axvline(x=cnt,ymin=0,ymax=511, color='red')
            plt.axhline(y=cnt, xmin=0, xmax=511, color='red')
            # for jj  in range(len(tt)):
            #     corr_sorted[ii] = tt[jj]
            #     ii += 1
        for jj in range(flen):
            newFeatList[jj] = torch.transpose(featurelist,0,1)[x[jj]]
        corrArray = np.corrcoef(newFeatList.cpu().numpy())

        plt.matshow(abs(corrArray), interpolation='nearest',cmap=plt.cm.Blues)

        cnt = 0
        for j in range(num_clusters):

            for jj in range(flen):
                if kmeans.labels_[jj] == j:

                    cnt += 1
            plt.axvline(x=cnt,ymin=0,ymax=511, color='red')
            plt.axhline(y=cnt, xmin=0, xmax=511, color='red')


        plt.show()
        pca = PCA(2)
        df = pca.fit_transform(corrArray)
        u_labels = np.unique(kmeans.labels_)
        for i in u_labels:
            plt.scatter(df[kmeans.labels_ == i, 0], df[kmeans.labels_ == i, 1], label=i)
            # plt.scatter(np.mean(df[kmeans.labels_ == i,0],np.mean(df[kmeans.labels_ == i,1])))
            # order of weights
            # plt.text(np.mean(df[kmeans.labels_ == i,0]),np.mean(df[kmeans.labels_ == i,1]),str(np.round(np.mean(orderperclass[kmeans.labels_==i].cpu().numpy()),0)))
            # mean corr of groups
            # plt.text(np.mean(df[kmeans.labels_ == i, 0]), np.mean(df[kmeans.labels_ == i, 1]),
            #      str(np.round(np.mean(corrArray[0][kmeans.labels_==i]),3)))
            #size of groups
            plt.text(np.mean(df[kmeans.labels_ == i, 0]), np.mean(df[kmeans.labels_ == i, 1]),
                 str(len(corrArray[0][kmeans.labels_ == i])))
        j = 0
        k = 0
        plt.clf()
        orderLabels = np.arange(11)
        orderlabelfeat = np.zeros(flen)
        for i in range(len(df)):
            if j > 50:
                k += 1
                j = 0
            orderlabelfeat[orderperclass[i]] = k
            j += 1
        u_labels = np.unique(orderlabelfeat)

        for i in u_labels:

            plt.scatter(df[orderlabelfeat == i, 0], df[orderlabelfeat == i, 1], label=int(i))


        plt.legend()
        plt.show()


        break

# for j in range(len(np.transpose(features))):
            #     fig, axs = plt.subplots(10 + 10*self.overclass, 1)
            #     fig.set_size_inches(18.5, 10.5, forward=True)
            #     if j == 10:
            #         break
                # for i in range(len(np.transpose(features))):
                #
                #     x = features[all_tar.cpu().numpy()==j].flatten()
                #     # scoresperClass = features[all_tar.cpu().numpy()==j][i]
                #     scoresperClass = x[i::10+10*self.overclass]
                #     axs[i].hist(scoresperClass, bins=200,range=(np.min(features[all_tar.cpu().numpy()==j])-0.5,np.max(features[all_tar.cpu().numpy()==j])+0.5))


            # plt.show()
            #     s = self.address+ 'hist target '+ str(j)
                # axs.set_title(s)
                # plt.savefig(s)