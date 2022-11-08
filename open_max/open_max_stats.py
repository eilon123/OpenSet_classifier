import matplotlib.pyplot as plt
import torch

from tsne import *

from open_max.openmax_utils import *
from open_max.evt_fitting import weibull_tailfitting, query_weibull

import numpy as np
import libmr
import csv
import matplotlib.patches as mpatches
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
        Whole = True

        for idx, layer in enumerate(self.layers):
            if idx <1 and self.useKmeans:
                continue

            begin = True
            targets = []
            if self.overclass:
                twoclass=True
                oneClass=False
            else:
                oneClass = True
                twoclass=False

            classIdx = os.listdir(os.path.join(self.feature_dir,"val", self.layers[0]))[0]
            classIdx = int(classIdx) - int(classIdx)%2
            classIdx = 4
            if self.overclass:
                classWhole = np.array([str(classIdx),str(classIdx+1)])
            else:
                classWhole = np.array(str(classIdx))
            i=0


            first=1
            for data in data_arr:

                for class_no in os.listdir(os.path.join(self.feature_dir,data, self.layers[0])):

                    featurefile_list = os.listdir(os.path.join(self.feature_dir,data, self.layers[0], class_no))


                    if self.useKmeans:
                        if data != 'open_set':
                            address = os.path.join(self.kmeans_path, layer, class_no) + '.pkl'
                            predictor = pickle.load(open(address, "rb"))
                    if data == 'val' and first and self.overclass:
                        first =0
                        classWhole = np.array([str(int(classIdx/2))])

                    for featurefile in featurefile_list:

                        scores = torch.from_numpy(np.load(os.path.join(self.feature_dir, data, self.layers[0], class_no, featurefile)))
                        newFeat = torch.from_numpy(np.load(os.path.join(self.feature_dir,data, layer, class_no, featurefile)))
                        if newFeat.shape[0] != 1:
                            newFeat = torch.reshape(newFeat, (1, newFeat.shape[0]))
                        if data == 'open_set' and self.useKmeans :
                            address = os.path.join(self.kmeans_path, layer, str(np.argmax(scores).item())) + '.pkl'
                            predictor = pickle.load(open(address, "rb"))
                        if begin:
                            begin = False
                            featurelist = torch.empty(size=(0,newFeat.shape[1]))
                        if i%500 ==0:
                            if i!=0:
                                featurelist = torch.cat((featurelist, tmplist), dim=0)
                            tmplist = torch.zeros(size=(500, newFeat.shape[1]))
                            k=0

                        tmplist[k] = newFeat
                        k+=1
                        # featurelist = torch.cat((featurelist, newFeat), dim=0)

                        if self.overclass and self.notSubclass==0:
                            cls = torch.argmax(scores)
                        elif self.useKmeans:

                            # cls = int(class_no)*(1+self.overclass)*self.num_clusters + predictor.predict(newFeat)
                            cls = int(class_no)+ predictor.predict(newFeat)
                            if predictor.predict(newFeat) ==2:
                                dbg=3

                        else:
                            cls = class_no
                        if Whole:
                            oset=2
                            val = 1
                            train = 0
                            if self.useKmeans:
                                train += predictor.predict(newFeat)
                                val = 2 + predictor.predict(newFeat)
                                oset = 4 + predictor.predict(newFeat)
                            elif self.overclass:
                                train += torch.argmax(scores).item() - classIdx
                                val = 2 + int(np.ceil(torch.argmax(scores).item() / 2) - int(classWhole[0]))
                                oset = 4 + int(np.ceil(torch.argmax(scores).item() / 2) - int(classWhole[0]))

                            if data == 'open_set' and str(int(torch.argmax(scores).item()/(1+self.overclass))) in(classWhole):
                                targets.append(oset)
                            elif data =='val' and class_no in classWhole:
                                targets.append(val)
                            elif data =='train' and class_no in classWhole:
                                targets.append(train)
                            else:
                                targets.append(-1)
                        else:
                            targets.append(int(cls))
                        i += 1
                if Whole==0:
                    break
            featurelist = torch.cat((featurelist, tmplist), dim=0)

            if self.useKmeans and idx ==0:
                x=3
            elif Whole:
                showtsneOneclass(featurelist.numpy(), targets, self.weibull_address + ' ' + layer, 'whole')
            else:
                showtsne(featurelist.numpy(), targets, self.weibull_address + ' ' + layer, data_type)

    def getVAR(self):
        add = self.weibull_address + '/var_test' + '.csv'
        csv_file = open(add, 'w')
        csv_val = csv.writer(csv_file, delimiter=",")

        add = self.weibull_address + '/var_open_set' + '.csv'
        csv_file = open(add, 'w')
        csv_open_set = csv.writer(csv_file, delimiter=",")
        data = 'val'
        n_digits = 10000
        for idx, layer in enumerate(self.layers):
            if idx <1 and self.useKmeans:
                continue
            cls_var = []
            for class_no in os.listdir(os.path.join(self.feature_dir, data, self.layers[0])):

                featurefile_list = os.listdir(os.path.join(self.feature_dir, data, self.layers[0], class_no))
                idx = 0
                if self.useKmeans:
                    address = os.path.join(self.kmeans_path, layer, class_no) + '.pkl'
                    predictor = pickle.load(open(address, "rb"))
                distl = 0
                cnt = 0
                cluster1 = []
                cluster2 = []
                featlist = []
                print(class_no)
                for featurefile in featurefile_list:

                    cnt+=1
                    Feat = torch.from_numpy(
                        np.load(os.path.join(self.feature_dir, data, layer, class_no, featurefile)))
                    Feat = torch.reshape(Feat, (1, Feat.shape[0]))
                    if self.useKmeans:
                        if predictor.predict(Feat) ==0:
                            cluster1.append(Feat)
                        else:
                            cluster2.append(Feat)
                    else:
                        featlist.append(Feat)
                if self.useKmeans:
                    if len(cluster1) == 0:
                        cluster1.append(torch.Tensor(0))
                    if len(cluster2) == 0:
                        cluster2.append(torch.Tensor(0))
                    cls_var.append((np.round(torch.mean(torch.var(torch.stack(cluster1), dim=0)).numpy(), 4),
                                    (np.round(torch.mean(torch.var(torch.stack(cluster2), dim=0)).numpy(), 4))))
                else:
                    cls_var.append((np.round(torch.mean(torch.var(torch.stack(featlist), dim=0)).numpy(), 4)))
                # print(distl/cnt)
            csv_val.writerow(cls_var)
        for idx, layer in enumerate(self.layers):
            if idx <1 and self.useKmeans:
                continue
            cls_var = []

            for class_closed in os.listdir(os.path.join(self.feature_dir, data, self.layers[0])):
                idx = 0
                print(class_closed)
                distl = 0
                cnt = 0
                cluster1 = []
                cluster2 = []
                featlist = []
                if self.useKmeans:
                    address = os.path.join(self.kmeans_path, layer, class_closed) + '.pkl'
                    predictor = pickle.load(open(address, "rb"))
                for class_no in os.listdir(os.path.join(self.feature_dir, 'open_set', self.layers[0])):
                    featurefile_list = os.listdir(os.path.join(self.feature_dir, 'open_set', self.layers[0], class_no))

                    for featurefile in featurefile_list:

                        cnt += 1
                        scores = torch.from_numpy(
                            np.load(os.path.join(self.feature_dir,  'open_set', self.layers[0], class_no, featurefile)))
                        if int(class_closed) != np.argmax(scores):
                            continue
                        Feat = torch.from_numpy(
                            np.load(os.path.join(self.feature_dir,  'open_set', layer, class_no, featurefile)))
                        Feat = torch.reshape(Feat, (1, Feat.shape[0]))
                        if self.useKmeans:
                            if predictor.predict(Feat) == 0:
                                cluster1.append(Feat)
                            else:
                                cluster2.append(Feat)
                        else:
                            featlist.append(Feat)
                if self.useKmeans:
                    if len(cluster1) == 0:
                        cluster1.append(torch.Tensor(0))
                    if len(cluster2) == 0 :
                        cluster2.append(torch.zeros(0))
                    cls_var.append((np.round(torch.mean(torch.var(torch.stack(cluster1), dim=0)).numpy(), 4),
                                    (np.round(torch.mean(torch.var(torch.stack(cluster2), dim=0)).numpy(), 4))))
                else:
                    cls_var.append((np.round(torch.mean(torch.var(torch.stack(featlist), dim=0)).numpy(), 4)))

            csv_open_set.writerow(cls_var)
    def getKmeansStats(self):
        if self.useKmeans==0:
            return

        data_arr = np.array(['train', 'val'])
        add = self.weibull_address + '/kmeans_train_num' + '.csv'
        csv_file = open(add, 'w')
        csv_train = csv.writer(csv_file, delimiter=",")
        add = self.weibull_address + '/kmeans_val_num' + '.csv'
        csv_file = open(add, 'w')
        csv_val = csv.writer(csv_file, delimiter=",")
        add = self.weibull_address + '/kmeans_open_set_num' + '.csv'
        csv_file = open(add, 'w')
        csv_open_set = csv.writer(csv_file, delimiter=",")
        for idx, layer in enumerate(self.layers):
            if idx <1 :
                continue
            for data in data_arr:
                cls_res = []
                break
                for class_no in os.listdir(os.path.join(self.feature_dir, data, self.layers[0])):

                    featurefile_list = os.listdir(os.path.join(self.feature_dir, data, self.layers[0], class_no))
                    idx = 0

                    address = os.path.join(self.kmeans_path, layer, class_no) + '.pkl'
                    predictor = pickle.load(open(address, "rb"))
                    distl = 0
                    cnt = 0
                    cluster1 = []
                    cluster2 = []
                    for featurefile in featurefile_list:
                        print(cnt)
                        cnt+=1
                        Feat = torch.from_numpy(
                            np.load(os.path.join(self.feature_dir, data, layer, class_no, featurefile)))
                        Feat = torch.reshape(Feat, (1, Feat.shape[0]))

                        if predictor.predict(Feat) ==0:
                            cluster1.append(Feat)
                        else:
                            cluster2.append(Feat)
                    cnt=0
                    cls_res.append((len(cluster1)*100/(1000*(1+4*int(data == 'train'))),len(cluster2)*100/(1000*(1+4*int(data == 'train')))))
                    continue
                    for feat in cluster1:
                        cnt += 1

                        print(cnt)
                        for feat2 in cluster2:
                            idx += 1

                            distl += compute_distance(self.ChooseFeat(feat, -1), self.ChooseFeat(feat2, -1))

                    cls_res.append(distl/idx)
                    print(cnt)
                    print(cls_res)
                    # print(distl/cnt)
                if data == 'val':
                    csv_val.writerow(cls_res)
                else:

                    csv_train.writerow(cls_res)

        for idx, layer in enumerate(self.layers):
            if idx <1:
                continue
            cls_res = []
            for class_closed in os.listdir(os.path.join(self.feature_dir, data, self.layers[0])):
                for class_no in os.listdir(os.path.join(self.feature_dir, 'open_set', self.layers[0])):
                    featurefile_list = os.listdir(os.path.join(self.feature_dir, 'open_set', self.layers[0], class_no))

                    idx = 0

                    distl = 0
                    cnt = 0
                    cluster1 = []
                    cluster2 = []
                    num=0
                    address = os.path.join(self.kmeans_path, layer, class_closed) + '.pkl'
                    predictor = pickle.load(open(address, "rb"))

                    print(class_no)
                    for featurefile in featurefile_list:

                        cnt += 1
                        scores = torch.from_numpy(
                            np.load(os.path.join(self.feature_dir, 'open_set', self.layers[0], class_no, featurefile)))
                        if int(class_closed) != np.argmax(scores):
                            continue
                        num+=1
                        Feat = torch.from_numpy(
                            np.load(os.path.join(self.feature_dir, 'open_set', layer, class_no, featurefile)))
                        Feat = torch.reshape(Feat, (1, Feat.shape[0]))
                        if predictor.predict(Feat) == 0:
                            cluster1.append(Feat)
                        else:
                            cluster2.append(Feat)
                cnt = 0
                if False:
                    cls_res.append((len(cluster1)*100/num, len(cluster2) *100/num))
                    continue
                for feat in cluster1:
                    cnt += 1
                    for feat2 in cluster2:
                        idx += 1

                        distl += compute_distance(self.ChooseFeat(feat, -1), self.ChooseFeat(feat2, -1))
                if distl == 0:
                    cls_res.append(0)
                else:
                    cls_res.append(distl/(cnt))
            csv_open_set.writerow(cls_res)

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
        plt.show()
        csv_file.close()
    def getTriDists(self):

        data_type = 'val'
        for idx, layer in enumerate(self.layers):
            print(idx)
            if idx == 3:
                dbg =3
            csvad = self.weibull_address + '/tridists' + layer +  '.csv'
            csv_file = open(csvad, 'w')
            csv_writer = csv.writer(csv_file, delimiter=",")
            row = ["test2mean","train2mean","NN"]
            csv_writer.writerow(row)
            img_features = np.load(os.path.join(self.feature_dir, 'val', layer, '0', '5001.npy'))
            scoremat, _ = self.BuildfeatMatrix('train', 'scores', self.Nclasses, 10)
            trainmat,_ = self.BuildfeatMatrix('train', layer, self.Nclasses, len(img_features))
            if data_type == 'val' or data_type == 'aug':
                numclasses = self.Nclasses
            elif data_type == 'open_set':
                numclasses = 4
            testMat,testtar = self.BuildfeatMatrix(data_type, layer, numclasses, len(img_features))
            osMat, ostar = self.BuildfeatMatrix('open_set', layer, numclasses, len(img_features))
            classChose = '3'
            meantrain_vec = np.load(os.path.join(self.mean_path, layer, classChose + ".npy"))
            i =0
            k=0
            testNN ,testNNidx= self.minDist(testMat, trainmat, numclasses ,0)
            osNN ,osNNidx= self.minDist(osMat, trainmat, numclasses,0)
            while i < 20:
                if i == 10:
                    row = ["openset"]
                    csv_writer.writerow(row)
                    data_type = 'open_set'
                    testMat = osMat
                    testtar = ostar
                    testNN = osNN
                    testNNidx = osNNidx
                if data_type == 'val':
                    if int(testtar[k]) != int(classChose):
                        k+=1
                        continue
                else:
                    if np.argmax(scoremat[k]) != int(classChose):
                        k+=1
                        continue
                k+=1
                row = []

                testmeandist = compute_distance(self.ChooseFeatLayer(testMat[k], int(classChose), layerType=layer), meantrain_vec,
                                        distance_type=self.distance_type)
                trainmeandist = compute_distance(self.ChooseFeatLayer(trainmat[testNNidx[k]], int(classChose), layerType=layer),
                                                meantrain_vec,
                                                distance_type=self.distance_type)
                row.append(testmeandist)
                row.append(trainmeandist)
                row.append(testNN[k])
                csv_writer.writerow(row)
                if testmeandist > trainmeandist + testNN[k]:
                    print("failure")
                i += 1


    def getNNDists(self):
        data_arr = np.array(['val', 'open_set'])
        osmindist = []
        mindist = []
        oneclass = 0
        clsnum=0
        for data_type in data_arr:
            # # print(layer)
            # img_features = np.load(os.path.join(self.feature_dir, 'val', self.layers[0], '0', '5001.npy'))
            # if oneclass:
            #     trainmat, _ = self.BuildfeatMatrix('train', self.layers[0], 1, len(img_features), clsnum='2')
            # else:
            #     trainmat, _ = self.BuildfeatMatrix('train', self.layers[0], self.Nclasses, len(img_features))
            # if data_type == 'val' or data_type == 'aug' or data_type == 'train':
            #     numclasses = self.Nclasses
            # elif data_type == 'open_set':
            #     numclasses = 4
            # if oneclass:
            #     numclasses = 1
            #     clsnum = '2'
            # Featmat, _ = self.BuildfeatMatrix(data_type, self.layers[0], numclasses, len(img_features), clsnum=clsnum)
            # if data_type == 'val':
            #     mdist, _ = self.minDist(Featmat, trainmat, numclasses, data_type == 'aug', Ntest=6)
            #
            # else:
            #     mdist,_ = self.minDist(Featmat, trainmat, numclasses, data_type == 'aug', Ntest=4)

            for idx, layer in enumerate(self.layers):
                print(layer)
                img_features = np.load(os.path.join(self.feature_dir, 'val', layer, '2', '0.npy'))
                if oneclass:
                    trainmat, _ = self.BuildfeatMatrix('train', layer, 1, len(img_features), clsnum='2')
                else:
                    trainmat, _ = self.BuildfeatMatrix('train', layer, self.Nclasses, len(img_features))
                if data_type == 'val' or data_type == 'aug' or data_type == 'train':
                    numclasses = self.Nclasses
                elif data_type == 'open_set':
                    numclasses = 4
                if oneclass:
                    numclasses = 1
                    clsnum = '2'
                Featmat, _ = self.BuildfeatMatrix(data_type, layer, numclasses, len(img_features),
                                                  clsnum=clsnum)
                if data_type == 'val':
                    mdist, _ = self.minDist(Featmat, trainmat, numclasses, data_type == 'aug', Ntest=6)
                    mindist.append(mdist)
                else:
                    mdist, _ = self.minDist(Featmat, trainmat, numclasses, data_type == 'aug', Ntest=4)
                    osmindist.append(mdist)



        xaxes = 'NN distance'
        yaxes = 'num of elements'
        titles = ['scores', 'L-1', 'L-2', 'L-3', 'L-4', 'L-5']

        f, a = plt.subplots(3, 2)
        a = a.ravel()
        for idx, ax in enumerate(a):

            x, bins, p = ax.hist(mindist[idx], density=True,label = str(data_arr[0]) )
            # if numclasses > 1:
            #     for item in p:
            #         item.set_height(item.get_height() / sum(x))
            ax.legend(loc='best')
            ax.set_title(titles[idx])
            ax.set_xlabel(xaxes)
            ax.set_ylabel(yaxes)
        f, b = plt.subplots(3, 2)
        b = b.ravel()
        for idx, ax in enumerate(b):
            x, bins, p = ax.hist(osmindist[idx], density=True, label=str(data_arr[1]))
            # for item in p:
            #     item.set_height(item.get_height() / sum(x))
            ax.legend(loc='best')
            ax.set_title(titles[idx])
            ax.set_xlabel(xaxes)
            ax.set_ylabel(yaxes)
        plt.tight_layout()
        plt.show()

    def getPCAdists(self):
        distTot = []
        trainTot = []
        predictor = list()
        for layernum in range(1, 4):
            tmp = []
            for i in range(self.Nclasses):
                address = os.path.join(self.pca_path, self.layers[layernum], str(i)) + '.pkl'
                tmp.append(pickle.load(open(address, "rb")))
            predictor.append(tmp)
        data_arr = np.array(['train', 'val'])
        for data_type in data_arr:

            distLayer = []
            for class_no in os.listdir(os.path.join(self.feature_dir, data_type, self.layers[0])):
                featurefile_list = os.listdir(os.path.join(self.feature_dir, data_type, self.layers[0], class_no))
                mean_path = os.path.join(self.mean_path)
                classChose = '0'
                if classChose != class_no:
                    continue
                meantrain_vec = []
                for i in range(1, 4):

                    meantrain_vec.append(np.load(os.path.join(mean_path, self.layers[i], class_no + ".npy")))
                for featurefile in featurefile_list:
                    Feat = []
                    scores = torch.from_numpy(
                        np.load(os.path.join(self.feature_dir, data_type, self.layers[0], class_no, featurefile)))
                    for i in range(1, 4):
                        Feat.append(np.load(
                            os.path.join(self.feature_dir, data_type, self.layers[i], class_no, featurefile)))

                    distance = self.pcadist(meantrain_vec, Feat, torch.argmax(scores),predictor)

                    distLayer.append(distance)

            if data_type == 'train':
                trainTot.append(distLayer)
            else:
                distTot.append(distLayer)
        osdist = []
        data_type = 'open_set'
        for idx, layer in enumerate(self.layers):
            distLayer = []

            for class_no in os.listdir(os.path.join(self.feature_dir, data_type, self.layers[0])):
                featurefile_list = os.listdir(os.path.join(self.feature_dir, data_type, layer, class_no))
                classChose = '0'
                mean_path = os.path.join(self.mean_path)
                meantrain_vec = []
                mean_path = os.path.join(self.mean_path)
                # mean_path = np.load(os.path.join(self.mean_path, layer, classChose + ".npy"))
                for i in range(1, 4):
                    meantrain_vec.append(np.load(os.path.join(mean_path, self.layers[i], classChose + ".npy")))
                for featurefile in featurefile_list:
                    scores = torch.from_numpy(
                        np.load(os.path.join(self.feature_dir, data_type, self.layers[0], class_no, featurefile)))
                    if np.argmax(scores) != int(classChose):
                        continue
                    Feat = []
                    for i in range(1, 4):
                        Feat.append(np.load(
                            os.path.join(self.feature_dir, data_type, self.layers[i], class_no, featurefile)))

                    distance = self.pcadist(meantrain_vec, Feat, torch.argmax(scores),predictor)

                    distLayer.append(distance)

            osdist.append(distLayer)

        xaxes = 'distance'
        yaxes = 'num of elements'
        titles = ['L-1 to L-3']

        if self.distance_type == 'eucos' or self.distance_type == 'cosine':
            distance = np.linspace(0, 2, 10000)
        else:
            distance = np.linspace(0, 30, 1000)
        x, bins, p = plt.hist(distTot[0], density=True,label = str(data_arr[1]) )
        for item in p:
            item.set_height(item.get_height() / sum(x))

        x, bins, p = plt.hist(osdist[0], density=True, label='open set')
        for item in p:
            item.set_height(item.get_height() / sum(x))
        mr = libmr.MR()
        tailtofit = sorted(trainTot[0])[-20:]
        mr.fit_high(tailtofit, len(tailtofit))
        wscore = []
        for dist in distance:
            wscore.append(mr.w_score(dist))
        plt.plot(distance, wscore, lw=2,label = 'weibull curve')
        plt.legend(loc='best')
        plt.title(titles[0])
        plt.xlabel(xaxes)
        plt.ylabel(yaxes)

        plt.tight_layout()
        plt.show()
    def getDists(self):
        distTot = []
        trainTot = []

        data_arr = np.array(['train','val'])
        for data_type in data_arr:
            for idx, layer in enumerate(self.layers):
                distLayer = []
                for class_no in os.listdir(os.path.join(self.feature_dir, data_type, self.layers[0])):
                    featurefile_list = os.listdir(os.path.join(self.feature_dir, data_type,layer, class_no))
                    classChose = '2'
                    meantrain_vec = np.load(os.path.join(self.mean_path, layer, classChose + ".npy"))
                    if self.overclass and data_type == 'val':
                        classChose = '1'
                    if classChose != class_no:
                        continue

                    for featurefile in featurefile_list:
                        Feat = torch.from_numpy(
                            np.load(os.path.join(self.feature_dir, data_type, layer, class_no, featurefile)))
                        print(featurefile)
                        dist = compute_distance(self.ChooseFeatLayer(Feat, int(class_no),layerType=layer), meantrain_vec,
                                 distance_type=self.distance_type)

                        distLayer.append(dist)

                    if data_type != 'open_set':
                        break
                if data_type == 'train':
                    trainTot.append(distLayer)
                else:
                    distTot.append(distLayer)
        osdist = []
        data_type = 'open_set'
        for idx, layer in enumerate(self.layers):
            distLayer = []


            for class_no in os.listdir(os.path.join(self.feature_dir, data_type, self.layers[0])):
                featurefile_list = os.listdir(os.path.join(self.feature_dir, data_type,layer, class_no))
                classChose='2'
                meantrain_vec = np.load(os.path.join(self.mean_path ,layer, classChose + ".npy"))
                for featurefile in featurefile_list:

                    Feat = torch.from_numpy(
                        np.load(os.path.join(self.feature_dir, data_type, layer, class_no, featurefile)))


                    scores = torch.from_numpy(
                        np.load(os.path.join(self.feature_dir, data_type, self.layers[0], class_no, featurefile)))
                    if np.argmax(scores) != int(classChose):
                        continue

                    dist = compute_distance(self.ChooseFeatLayer(Feat, int(classChose), layerType=layer),
                                            meantrain_vec,
                                            distance_type=self.distance_type)

                    distLayer.append(dist)

            if idx==3:
                x=3
            osdist.append(distLayer)
        xaxes = 'distance'
        yaxes = 'num of elements'
        titles = ['scores', 'L-1', 'L-2', 'L-3','L-4','L-5']

        f, a = plt.subplots(3, 2)
        a = a.ravel()
        if self.distance_type == 'eucos' or self.distance_type == 'cosine':
            distance = np.linspace(0, 2, 10000)
        else:
            distance = np.linspace(0, 30, 1000)
        for idx, ax in enumerate(a):

            # ax.hist(distTot[idx],density=1,label = str(data_arr[1]) )
            x, bins, p = ax.hist(distTot[idx], density=True,label = str(data_arr[1]) )
            for item in p:
                item.set_height(item.get_height() / max(x))
            # ax.hist(osdist[idx], density=1,label = 'open set')
            x, bins, p = ax.hist(osdist[idx], density=True, label='open set')
            for item in p:
                item.set_height(item.get_height() / max(x))
            mr = libmr.MR()
            tailtofit = sorted(trainTot[idx])[-20:]
            mr.fit_high(tailtofit, len(tailtofit))
            wscore = []
            for dist in distance:
                wscore.append(mr.w_score(dist))
            ax.plot(distance, wscore, lw=2,label = 'weibull curve')
            ax.legend(loc='best')
            ax.set_title(titles[idx])
            ax.set_xlabel(xaxes)
            ax.set_ylabel(yaxes)
            ax.set_ylim([0,1.2])
        plt.tight_layout()
        plt.show()
    def getAcc(self):
        cnt = 0
        for cls_no in os.listdir(os.path.join(self.feature_dir, 'val', self.layerType)):
            for filename in os.listdir(os.path.join(self.feature_dir, 'val', self.layerType, cls_no)):
                img_features = np.load(os.path.join(self.feature_dir, 'val', self.layerType, cls_no, filename))
                ddmin = 1000000
                for ii in range(self.Nclasses):
                    dd = self.distToMAV(str(ii), img_features)
                    if dd < ddmin:
                        ddmin = dd
                        cls_pred = ii
                if cls_pred == int(cls_no):
                    cnt += 1
        print(cnt * 100 / (self.Nclasses * 1000))
# k = category_weibull[2].get_params()[1]
# lamda = category_weibull[2].get_params()[0]