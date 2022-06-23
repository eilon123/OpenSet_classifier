
import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
import torch
from scipy.io import loadmat
from numpy import linalg as LA
from tsne import *

from open_max.openmax_utils import *
from open_max.evt_fitting import weibull_tailfitting, query_weibull

import numpy as np
import libmr
import scipy.stats as st
import csv
from sklearn.metrics import roc_auc_score
import random
import torch.nn as nn
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.integrate as integrate
from open_max.open_max_father import *


red_patch = mpatches.Patch(color='red', label='same class')
green_patch_False = mpatches.Patch(color='green', label='misstaken class')
black_patch_False = mpatches.Patch(color='black', label='misstaken class')

green_patch = mpatches.Patch(color='green', label='diff class')
blue_patch = mpatches.Patch(color='blue', label='diff class')


class openmax_stats(openmax):
    def __init__(self, args, open_max_t,save_path=""):
        super().__init__(args, open_max_t)
        self.feature_dir = "open_max/saved_features/cifar10/"

        self.distance_path = "open_max/saved_distance_scores/cifar10/"

        self.mean_path = "open_max/saved_MAVs/cifar10/"
        self.kmeans_path = "open_max/saved_kmeans/cifar10/"
        self.classes = []
        self.pca_path = "open_max/saved_pca/cifar10/"
        self.alpha_rank = int(self.Nclasses/(1+self.overclass))
        self.weibull_tailsize = 20


        self.save_path = save_path
        networktype = 'regular'
        if self.overclass:
            networktype = 'overclass'
        self.weibull_address = 'open_max/weibull/' + self.save_path + networktype

        self.distance_type = 'euclidean'
        os.makedirs(os.path.join(self.weibull_address), exist_ok=True)
        if self.distance_type == 'eucos' or self.distance_type == 'cosine':
            self.dist = np.linspace(0, 2, 10000)
        else:
            self.dist = np.linspace(0, 30, 1000)
    def routine(self):

        mean_path = os.path.join(self.mean_path, self.layerType)
        distance_path = os.path.join(self.distance_path, self.layerType)
        # self.weibull_model = weibull_tailfitting(mean_path, distance_path, useKmeans=self.useKmeans,
        #                                          num_clusters=self.num_clusters,
        #                                          tailsize=self.weibull_tailsize, distance_type=self.distance_type)
        tailSize = np.array([5, 10, 20, 30, 50, 100, 200, 500])
        w = list()

        for tail in tailSize:
            w.append(
                weibull_tailfitting(mean_path, distance_path, useKmeans=self.useKmeans, num_clusters=self.num_clusters,
                                    tailsize=tailSize, distance_type=self.distance_type,oneClass=True))
        prec, opensetprec = self.getCDF("val")
        self.get_score_table("val")
        self.get_score_table("open_set")


    def get_score_table(self,data_type='val'):
        for i in range(-1,3):
            self.setAddress(i)

            s1 = self.weibull_address  +  "/scores_table_" + data_type + self.layerType +".csv"
            s2 = self.weibull_address  + "/var_table_" + data_type + self.layerType +".csv"
            csv_file = open(s1, 'w')
            var_file = open(s2, 'w')

            csv_writer = csv.writer(csv_file, delimiter=",")
            var_writer = csv.writer(var_file, delimiter=",")




            lst = os.listdir(os.path.join(self.feature_dir, data_type,self.layerType))
            lst.sort()
            num_classes = len(lst)

            for i in range(self.Nclasses):
                for cls_no in (lst):
                    scores = list()
                    for filename in os.listdir(os.path.join(self.feature_dir, data_type,self.layerType,( cls_no))):


                        img_features = np.load(os.path.join(self.feature_dir, data_type,self.layerType, (cls_no), filename))
                        if np.argmax(img_features) != i:
                            continue
                        scores.append(img_features[0:self.Nclasses])
                if len(scores) ==0:
                    scores.append(np.zeros(shape=self.Nclasses))
                arr = list()
                stdlist = list()
                for cls in range(self.Nclasses):
                    arr.append(0)
                    for lngt in range(len(scores)):
                        arr[cls] += scores[lngt][cls]
                    arr[cls] /= len(scores)
                    arr[cls] = round(arr[cls],1)
                    std = 0
                    for lngt in range(len(scores)):
                        std += (scores[lngt][cls] - arr[cls])**2
                    std /= len(scores)
                    std = np.sqrt(std)
                    std = round(std, 1)
                    stdlist.append(std)

                csv_writer.writerow(arr)
                var_writer.writerow(stdlist)


            csv_file.close()
            if self.deepclassifier==0:
                break
    def plotCDF(self,address,origscores,dist,scores,patch,cls_no,falsepts,falsescores):

        os.makedirs(address, exist_ok=True)

        plt.plot(self.dist, origscores)
        plt.scatter(dist, scores, color='red', marker='*')

        plt.legend(handles=[patch])
        d = "class num " + cls_no
        plt.title(d)
        s = os.path.join(address,str(cls_no))

        plt.plot(self.dist, scores)
        plt.scatter(falsepts, falsescores, color='black', marker='*')

    def getprecision(self,scores,length):
        x = np.array(scores)

        return (sum(x > 0.5) * 100 / (length+0.001))

    def getCDF(self, data_type, quick=0, cls_check=-1):

        src = os.path.join(self.feature_dir, data_type, self.layerType)
        csvad = self.weibull_address + '/acc_' + data_type + '.csv'
        csv_file = open(csvad, 'w')
        csv_writer = csv.writer(csv_file, delimiter=",")
        percTot = np.zeros(4)
        if data_type == 'train' or data_type == 'val':
            classes = os.listdir(src)
            classes.sort()
            for cls_no in classes:

                score = list()
                sameclasspoints = list()
                sameclassscore = list()
                falseclasspoints = list()
                falseclassscore = list()
                otherclasspoints = list()
                otherclassscore = list()
                category_weibull = query_weibull(cls_no, self.weibull_model, distance_type=self.distance_type)
                if self.overclass:
                    clist = list()
                    clist.append(query_weibull(cls_no, self.weibull_model, distance_type=self.distance_type))
                    clist.append(query_weibull(str(int(cls_no)+1), self.weibull_model, distance_type=self.distance_type))

                for filename in os.listdir(os.path.join(self.feature_dir, data_type, self.layerType, cls_no)):
                    img_features = np.load(
                        os.path.join(self.feature_dir, data_type, self.layerType, cls_no, filename))
                    for cls_num in os.listdir(src):

                        if int(cls_no) != int(cls_num):
                            if int(np.argmax(img_features)/(1+self.overclass)) != int(cls_no):
                                continue
                            # c = query_weibull(cls_num, w, distance_type=distance_type)
                            distance = compute_distance(img_features, category_weibull[0],
                                                        distance_type=self.distance_type)

                            wscore = category_weibull[2].w_score(distance)

                            otherclassscore.append(wscore)
                            otherclasspoints.append(distance)

                        else:
                            distance = compute_distance(img_features, category_weibull[0],
                                                        distance_type=self.distance_type)


                            wscore = category_weibull[2].w_score(distance)

                            if int(np.argmax(img_features)/(1+self.overclass)) == int(cls_no):
                                sameclasspoints.append(distance)
                                sameclassscore.append(wscore)
                            else:
                                falseclasspoints.append(distance)
                                falseclassscore.append(wscore)

                s = self.weibull_address + '/cdf/' + data_type + '/' + 'sameclass'

                os.makedirs(s, exist_ok=True)

                # plt.plot(self.dist, score)
                # plt.scatter(sameclasspoints, sameclassscore, color='red', marker='*')
                #
                # plt.legend(handles=[red_patch])
                # s = "class num " + cls_no
                # plt.title(s)
                #
                # s = self.weibull_address + '/cdf/' + data_type + '/' + 'sameclass/' + str(cls_no)
                #
                # plt.plot(self.dist, score)
                # plt.scatter(falseclasspoints, falseclassscore, color='black', marker='*')
                # self.plotCDF(self.weibull_address + '/cdf/' + data_type + '/' + 'sameclass',score, sameclasspoints, sameclassscore, red_patch)
                x = np.array(sameclassscore)
                y = np.array(sameclasspoints)
                lenlist = len(sameclasspoints) + 0.01
                perc = list()
                perc.append(np.round(sum(x > 0.9) * 100 / lenlist, 1))
                perc.append(sum(x > 0.5) * 100 / lenlist)
                x = np.array(falseclasspoints)
                perc.append(np.round(sum(x > 0.9) * 100 / lenlist, 1))
                x = np.array(otherclassscore)
                perc.append(np.round(sum(x > 0.9) * 100 / (len(otherclassscore) + 0.01), 1))
                for i in range(4):
                    percTot[i] += perc[i] / 10

                csv_writer.writerow(perc)

                # plt.legend(handles=[red_patch, black_patch_False])
                # plt.savefig(s)
                # plt.clf()
                # plt.plot(self.dist, score)
                # plt.scatter(otherclasspoints, otherclassscore, color='green')
                # s = "class num " + cls_no
                # plt.title(s)
                # plt.legend(handles=[green_patch])
                # s = self.weibull_address + '/cdf/' + data_type + '/' + 'diffclass'
                #
                # os.makedirs(s, exist_ok=True)
                # s = s + '/' + str(cls_no)
                # plt.savefig(s)
                # plt.clf()
                s = self.weibull_address + '/cdf/' + data_type + '/' + 'allData'

                os.makedirs(s, exist_ok=True)
                s = s + '/' + str(cls_no)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(self.dist, score)
                ax.scatter(falseclasspoints, falseclassscore, color='black', marker='*', label='misstaken')
                ax.scatter(sameclasspoints, sameclassscore, color='red', marker='*', label='same class')
                ax.scatter(otherclasspoints, otherclassscore, color='green', label='other class')
                opensetscore, opensetpoints, precision = self.getCDF('open_set', quick=1, cls_check=cls_no)
                ax.scatter(opensetpoints, opensetscore, color='orange', label='other class')
                # ax.plt.legend(handles=[red_patch, black_patch_False,green_patch])
                leg1 = ax.legend(loc='upper left')
                # Add second legend for the maxes and mins.
                # leg1 will be removed from figure
                st0 = ''
                st1 = '90% probability = ' + str(perc[0])
                st2 = 'misstaken 90% probability = ' + str(perc[4])
                st3 = 'other classes 90% probability = ' + str(perc[5])
                st4 = 'open set 90% probability ' + str(precision)
                leg2 = ax.legend([st0, st1, st2, st3, st4], loc='lower right')

                # s = 'preds over 90% = ' + str(perc[0])
                # plt.text(1, 1, s, style='italic',
                #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
                ax.add_artist(leg1)
                ax.add_artist(leg2)
                plt.savefig(s)

                plt.clf()
                if False:
                    return perc, precision
        else:
            print("dist of open set")
            src2 = os.path.join(self.feature_dir, 'open_set', self.layerType)
            direc = os.listdir(src)
            direc.sort()
            for cls_no in direc:
                score = list()
                # if quick and cls_no != cls_check:
                #     continue
                opensetpoints = list()
                opensetscore = list()
                cnt = 0
                total = 0
                category_weibull = query_weibull(cls_no, self.weibull_model, distance_type=self.distance_type)
                for d in dist:
                    score.append(category_weibull[2].w_score(d))
                for cls_open in os.listdir(src2):

                    # for cls_num in (os.listdir(os.path.join(feature_path, data_type))):
                    for filename in os.listdir(os.path.join(src2, cls_open)):

                        img_features = np.load(
                            os.path.join(self.feature_dir, data_type, self.layerType, cls_open, filename))
                        if cls_check:
                            if np.argmax(img_features) != int(cls_check):
                                continue
                        elif np.argmax(img_features) != int(cls_no):
                            continue
                        total += 1
                        img_features_cls = img_features[int(cls_no)]
                        # img_features_cls = img_features
                        distance = compute_distance(img_features[int(cls_no)], category_weibull[0],
                                                    distance_type=self.distance_type)
                        wscore = category_weibull[2].w_score(distance)
                        if wscore < 0.5:
                            cnt += 1
                        opensetscore.append(wscore)
                        opensetpoints.append(distance)
                x = np.array(opensetscore)
                y = np.array(opensetpoints)

                precision = (np.round(sum(x > 0.9) * 100 / (len(opensetpoints) + 0.01), 1))
                if quick:
                    return opensetscore, opensetpoints, precision
                s = self.weibull_address + '/cdf/' + data_type

                os.makedirs(s, exist_ok=True)
                plt.plot(dist, score)
                plt.scatter(opensetpoints, opensetscore, color='blue')
                s = "class num " + cls_no
                plt.title(s)
                plt.legend(handles=[blue_patch])
                s = self.weibull_address + '/cdf/' + data_type + '/' + str(cls_no)

                plt.savefig(s)
                plt.clf()

        csv_writer.writerow(percTot)
        csv_file.close()
        perc = 0
        precision = 0
        return perc, precision

    def getPDF(self, data_type="train"):
        if self.distance_type == 'eucos' or self.distance_type == 'cosine':
            distance = np.linspace(0, 2, 10000)
        else:
            distance = np.linspace(0, 10, 4000)

        totlist = list()
        dirlist = os.listdir(os.path.join(self.feature_dir, data_type))
        num_classes = len(os.listdir(os.path.join(self.feature_dir, "train")))
        dirlist.sort
        numClass = len(dirlist)

        for cls_indx in range(numClass * (1 + self.overclass)):
            cnt = 0
            wscore = list()
            distl = list()
            distl2 = list()
            distFalse = list()
            # meantrain_vec = np.load(os.path.join(mean_path, str(cls_indx) + ".npy"))
            category_weibull = query_weibull(cls_indx, self.weibull_model, distance_type=self.distance_type)
            k = category_weibull[2].get_params()[1]
            lamda = category_weibull[2].get_params()[0]

            for dist in distance:
                # wscore.append((k / lamda) * (dist / lamda) ** (k - 1) * np.exp(-(dist / lamda) ** k))

                wscore.append(category_weibull[2].w_score(dist) - category_weibull[2].w_score(dist - 0.01))

            fig, ax = plt.subplots()
            color = 'tab:orange'
            ax.plot(distance, wscore, color=color)
            ax.tick_params(axis='y', labelcolor=color)

            onelist = list()
            for cls_no in range(num_classes):

                if int(cls_no) != cls_indx:

                    for filename in os.listdir(os.path.join(self.feature_dir, data_type, str(cls_indx))):
                        img_features = np.load(os.path.join(self.feature_dir, data_type, str(cls_indx), filename))
                        c = query_weibull(cls_no, self.weibull_model, distance_type=self.distance_type)
                        tau = c[2].get_params()[4]
                        dil = compute_distance(img_features[int(cls_no)], c[0], distance_type=self.distance_type)

                        distl.append(
                            compute_distance(img_features[int(cls_no)], c[0], distance_type=self.distance_type))


                else:

                    for filename in os.listdir(os.path.join(self.feature_dir, data_type, str(cls_indx))):

                        c = query_weibull(cls_no, self.weibull_model, distance_type=self.distance_type)
                        img_features = np.load(os.path.join(self.feature_dir, data_type, str(cls_indx), filename))
                        tau = c[2].get_params()[4]
                        dd = compute_distance(img_features[int(cls_no)], c[0], distance_type=self.distance_type)

                        if np.argmax(img_features) == cls_indx:
                            if dd > 2.5:
                                cnt += 1
                            distl2.append(dd)
                        else:
                            distFalse.append(dd)

            print("cls i s ", cls_indx)
            print(100 - cnt * 100 / len(distl2))
            totlist.append(onelist)
            # plt.show()
            color = 'tab:blue'

            s = self.weibull_address + '/pdf/' + data_type

            os.makedirs(s, exist_ok=True)

            ax2 = ax.twinx()
            ax2.hist(distl, bins=40, density=True, color=color, label='other claseses')
            ax2.tick_params(axis='y', labelcolor=color)

            color = 'tab:red'
            ax3 = ax2.twinx()
            ax3.hist(distl2, bins=40, density=True, color=color, label='right classes')
            ax3.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()

            color = 'tab:green'
            ax4 = ax3.twinx()
            ax4.hist(distFalse, bins=40, density=True, color=color, label='misstaken')
            ax4.tick_params(axis='y', labelcolor=color)
            fig.tight_layout()
            #
            ax4.legend(handles=[red_patch, blue_patch, green_patch_False])
            plt.savefig(s)
            plt.clf()
            dist2 = np.linspace(0, 10000, 100000)
        print("finish")

    def get_csv(self):
        import csv
        totlist = list()
        s = self.weibull_address + "/distance_table.csv"
        csv_file = open(s, 'w')
        mean_path = os.path.join('open_max/saved_MAVs', 'cifar10')
        csv_writer = csv.writer(csv_file, delimiter=",")

        data_type = 'val'
        lst = os.listdir(os.path.join(self.feature_dir, data_type))
        lst.sort()
        for cls_indx in range(len(lst)):

            dist = list()

            meantrain_vec = np.load(os.path.join(mean_path, str(cls_indx) + ".npy"))
            category_weibull = query_weibull(cls_indx, self.weibull_model, distance_type=self.distance_type)
            mean = (round(meantrain_vec.item(), 2))
            dist.append(mean)

            for cls_no in lst:
                distl = list()
                for filename in os.listdir(os.path.join(self.feature_dir, data_type, cls_no)):
                    c = query_weibull(cls_no, self.weibull_model, distance_type=self.distance_type)
                    img_features = np.load(os.path.join(self.feature_dir, data_type, cls_no, filename))
                    # distl.append(compute_distance(img_features[0:5], category_weibull[0], distance_type=distance_type))
                    distl.append(compute_distance(img_features[int(cls_indx)], category_weibull[0],
                                                  distance_type=self.distance_type))
                    # distl.append(compute_distance(img_features[int(cls_indx)], c[0], distance_type=distance_type))

                meandist = round(np.mean(distl), 2)
                vardist = np.sqrt(round(np.var(distl), 3))

                dist.append((meandist))

            csv_writer.writerow(dist)  # rowtocsv.append(dist)
        # for row in rowtocsv:

        csv_file.close()

    def getTSNE(self, data_type='val'):


        data_arr = np.array(['train','val','open_set'])
        print(self.layers)

        for idx, layer in enumerate(self.layers):
            if idx ==0:
                continue
            begin = True
            targets = []
            if self.overclass:
                twoclass=True
                oneClass=False
            else:
                oneClass = True
                twoclass=False
            Whole = True

            targetWhole = []
            classWhole = os.listdir(os.path.join(self.feature_dir,"val", self.layers[0]))[0]
            for data in data_arr:
                if data == 'open_set':
                    x=3

                for class_no in os.listdir(os.path.join(self.feature_dir,data, self.layers[0])):
                    featurefile_list = os.listdir(os.path.join(self.feature_dir,data, self.layers[0], class_no))

                    i = 0
                    if self.useKmeans:

                        address = os.path.join(self.kmeans_path, layer, class_no) + '.pkl'
                        predictor = pickle.load(open(address, "rb"))
                    for featurefile in featurefile_list:

                        scores = torch.from_numpy(np.load(os.path.join(self.feature_dir, data, self.layers[0], class_no, featurefile)))
                        newFeat = torch.from_numpy(np.load(os.path.join(self.feature_dir,data, layer, class_no, featurefile)))
                        if newFeat.shape[0] != 1:
                            newFeat = torch.reshape(newFeat, (1, newFeat.shape[0]))

                        if begin:
                            begin = False
                            featurelist = torch.empty(size=(0,newFeat.shape[1]))

                        featurelist = torch.cat((featurelist, newFeat), dim=0)
                        if self.overclass and self.notSubclass==0:
                            cls = torch.argmax(scores)
                        elif self.useKmeans:

                            # cls = int(class_no)*(1+self.overclass)*self.num_clusters + predictor.predict(newFeat)
                            cls = int(class_no)   + predictor.predict(newFeat)
                            if predictor.predict(newFeat) ==2:
                                dbg=3

                        else:
                            cls = class_no
                        if Whole :
                            if data == 'open_set' and torch.argmax(scores).item() == int(classWhole):

                                targets.append(int(classWhole) +1)
                            elif data =='val' and classWhole == class_no:
                                targets.append(int(classWhole) + 2)
                            elif classWhole == class_no:
                                targets.append(int(cls))
                        elif Whole:
                            targets.append(0)
                        else:
                            targets.append(int(cls))
            if self.useKmeans and idx ==0:
                x=3
            elif Whole:
                showtsneOneclass(featurelist.numpy(), targets, self.weibull_address + ' ' + layer, 'whole')
            else:
                showtsne(featurelist.numpy(), targets, self.weibull_address + ' ' + layer, data_type)



    def getoneCDF(self, data_type='val'):
        mean_path = os.path.join(self.mean_path, self.layerType)
        distance_path = os.path.join(self.distance_path, self.layerType)
        tail = np.array([5,10,20,50,100,200,500])

        csvad = self.weibull_address + '/acc_per_weibull' + '.csv'
        csv_file = open(csvad, 'w')
        csv_writer = csv.writer(csv_file, delimiter=",")

        classIdx = '0'
        if data_type == 'train' or data_type == 'val':
            src = os.path.join(self.feature_dir, data_type, self.layerType)
            classes = os.listdir(src)
            classes.sort()
            plt.clf()
            st = []
            for i in range(len(tail)):
                weibull_model = weibull_tailfitting(mean_path, distance_path, useKmeans=self.useKmeans,
                                                         num_clusters=self.num_clusters,
                                                         tailsize=tail[i], distance_type=self.distance_type)
                category_weibull = query_weibull(classIdx, weibull_model, distance_type=self.distance_type)
                wscore = []
                for dist in self.dist:

                    wscore.append(category_weibull[2].w_score(dist))
                plt.plot(self.dist,wscore)
                classes = os.listdir(src)
                classes.sort()
                for cls_no in classes:
                    otherclassscore = []
                    sameclassscore = []
                    falseclassscore = []
                    opensetScore = []
                    src = os.path.join(self.feature_dir, data_type, self.layerType)
                    for filename in os.listdir(os.path.join(self.feature_dir, data_type, self.layerType, classIdx)):
                        img_features = np.load(
                            os.path.join(self.feature_dir, data_type, self.layerType, classIdx, filename))
                        for cls_num in os.listdir(src):

                            if int(classIdx) != int(cls_num):
                                if int(np.argmax(img_features)/(1+self.overclass)) != int(cls_num):
                                    continue

                                distance = compute_distance(self.ChooseFeat(img_features,cls_num), category_weibull[0],
                                                            distance_type=self.distance_type)

                                wscore = category_weibull[2].w_score(distance)

                                otherclassscore.append(wscore)

                            else:
                                distance = compute_distance(self.ChooseFeat(img_features,cls_num), category_weibull[0],
                                                            distance_type=self.distance_type)


                                wscore = category_weibull[2].w_score(distance)

                                if int(np.argmax(img_features)/(1+self.overclass)) == int(cls_no):
                                    sameclassscore.append(wscore)
                                else:
                                    falseclassscore.append(wscore)
                    break
                src2 = os.path.join(self.feature_dir, 'open_set', self.layerType)
                classes = os.listdir(src2)
                classes.sort()
                for classOpen in classes:
                    for filename in os.listdir(os.path.join(self.feature_dir, 'open_set', self.layerType, classOpen)):
                        img_features = np.load(
                            os.path.join(self.feature_dir, 'open_set', self.layerType, classOpen, filename))


                        if int(np.argmax(img_features) / (1 + self.overclass)) != int(classIdx):
                            continue

                        distance = compute_distance( self.ChooseFeat(img_features,cls_num), category_weibull[0],
                                                    distance_type=self.distance_type)

                        wscore = category_weibull[2].w_score(distance)

                        opensetScore.append(wscore)
                acc = list()
                acc.append(tail[i])
                acc.append(np.round(self.getprecision(sameclassscore,len(np.array(sameclassscore))),2))
                acc.append(np.round(self.getprecision(falseclassscore,len(np.array(falseclassscore))),2))
                acc.append(np.round(self.getprecision(otherclassscore,len(np.array(otherclassscore))),2))
                acc.append(np.round(self.getprecision(opensetScore,len(np.array(opensetScore))),2))
                csv_writer.writerow(acc)
        st1 = str(tail[0])
        st2 = str(tail[1])
        st3 = str(tail[2])
        st4 = str(tail[3])
        st5 = str(tail[4])
        st6 = str(tail[5])
        st7 = str(tail[6])
        leg2 = plt.legend([ st1, st2, st3, st4,st5,st6,st7], loc='lower right')
        # plt.show()
        csv_file.close()

# k = category_weibull[2].get_params()[1]
# lamda = category_weibull[2].get_params()[0]