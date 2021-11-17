
import  torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

import matplotlib.pyplot as plt
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
        predProb = F.softmax(outputs[i][extraClass * predicted[i].item():extraClass * predicted[i].item() + extraClass], dim=0)
        gradeHist[int(histIdx[predicted[i].item()])][extraClass * predicted[i].item()] = predProb[0]
        gradeHist[int(histIdx[predicted[i].item()])][extraClass * predicted[i].item() + 1] = predProb[1]
        histIdx[predicted[i].item()] += 1
    return gradeHist
def showHist(gradeHist, address ,numHist,union):
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
    plt.savefig(address + 'histogram')



