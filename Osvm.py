from sklearn.svm import OneClassSVM
import torch
import copy
from createDataset import *
from sklearn.decomposition import PCA
import wandb
import matplotlib.pyplot as plt
from utils import progress_bar
import math

m = nn.Softmax(dim=1)

class osvm():
    def __init__(self,args ,device,net,trainset,testloader,targets,address):
        super(osvm, self).__init__()
        self.net = net
        self.overclass = args.overclass
        self.takeFeat = True
        self.batch = args.batch
        self.trainset = trainset
        self.num_classes = 10
        self.classifiersNum = self.num_classes * (args.overclass +1)
        self.testloader = testloader
        wandb.watch(self.net)
        self.address = address

        self.device = device
        self.trans = args.trans
        self.targets = targets
        wandb.watch(self.net)

    def setTestloader(self,testloader,targets):
        self.testloader = testloader
        self.targets = targets
    def createoverClassDataSet(self):
        if not(self.overclass):
            return self.trainset
        self.net.eval()
        overClassdataSet = copy.deepcopy(self.trainset)
        if self.trans:

            classidx_to_keep = np.array([0,1,2,3,4])
            trainloader,outputSet = createDataset(self.trainset, classidx_to_keep, True, self.batch, test=False)

        else:
            trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=self.batch, shuffle=False, num_workers=8)


        with torch.no_grad():
            sum = 0
            tot = 0
            correct = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, pool, featureLayer, _ = self.net(inputs)

                _, predicted = outputs.max(1)

                correct += predicted.eq(targets).sum().item()
                predicted = predicted.cpu().numpy()
                for i,pred in enumerate(predicted):

                    tot +=1
                    if self.trans:
                        realTar = outputSet.targets[int(i + batch_idx * (trainloader.batch_size))]
                    else:
                        realTar = self.trainset.targets[int(i + batch_idx*(trainloader.batch_size))]
                    if int(pred.item()/2) != realTar:
                        predicted[i] = 2*realTar + (outputs[i][2*realTar + 1] > outputs[i][2*realTar])
                    else:
                        sum +=1
                overClassdataSet.targets[batch_idx*trainloader.batch_size:(batch_idx+1)*trainloader.batch_size] = predicted
                # progress_bar(i, self.classifiersNum)
        cor = sum / tot
        return overClassdataSet
    def getSupport(self,dataset):
        global best_acc
        self.net.eval()
        clfArr = list()
        pca = list()
        # self.classifiersNum = 4
        # self.overclass=1
        with torch.no_grad():
            for i in range(self.classifiersNum):
                trainloader,_ = createDataset(dataset,np.array([i]), isTrain=True,batchSize=self.batch,test=False)

                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs,pool, featureLayer,_ = self.net(inputs)
                    featureLayer = featureLayer.cpu().numpy()
                    outputs = outputs.cpu().numpy()
                    if self.takeFeat:
                        if batch_idx == 0:
                            features = featureLayer
                        else:
                            features = np.concatenate((features, featureLayer))
                    else:
                        if batch_idx == 0:
                            features = outputs
                        else:
                            features = np.concatenate((features, outputs))
                progress_bar(i, self.classifiersNum)
                clf = OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1*(1+7*self.overclass))
                # clf = OneClassSVM(nu=0.01, kernel="sigmoid", gamma='scale')

                # clf = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale')
                clf.fit(features)
                clfArr.append(clf)
                if False:
                    currPca=(PCA(n_components=2))
                    pca.append(currPca)
                    pca[i].fit(features)
                    eigenvalues = pca[i].explained_variance_

                    predicted = clfArr[i].predict(features)
                    index = np.where(predicted == -1)
                    featReduced = pca[i].transform(features)
                    values = featReduced[index]

                    plt.scatter(featReduced[:, 0], featReduced[:, 1])
                    plt.scatter(values[:, 0], values[:, 1], color='r')
                    plt.savefig(self.address[0:10]  + 'osvm class ' + str(i) + '.png')
                    plt.clf()
                    for j in range(len(featReduced)):
                        if i != 0:
                            featReduced = pca[0].transform(features)
                        c = 1/(math.sqrt(2*math.pi*(eigenvalues[0] + eigenvalues[1])))
                        scores.append(c * math.exp(-0.5 * (featReduced[j, 0] / eigenvalues[0] + featReduced[j, 1] / eigenvalues[1])))
                    # if i%2 == 1 and self.overclass and False:
                    #     predicted1 = clfArr[i-1].predict(features)
                    #     index = np.where(predicted1 == -1)
                    #
                    #     featReduced1 = pca[i-1].transform(features)
                    #     values1 = featReduced[index]
                    #     plt.scatter(featReduced[:, 0], featReduced[:, 1])
                    #     plt.scatter(values[:, 0], values[:, 1], color='r')
                    #     plt.scatter(featReduced1[:, 0], featReduced1[:, 1])
                    #     plt.scatter(values1[:, 0], values1[:, 1], color='g')
                    #     plt.savefig(self.address[0:10] + 'overclass osvm class  ' + str(i) + '.png')
                    #     plt.clf()
                    # # wandb.log({"variance": (pca. / self.trainBsize)})
                    # # wandb.log({mode: wandb.Image(address[0:13] + 'osvm class ' + str(i) + '.png')})
                    data = [[s] for s in scores]
                    table = wandb.Table(data=data, columns=["scores"])
                    wandb.log({'my_histogram': wandb.plot.histogram(table, "scores",
                        title ="class" + str(i))})

        return clfArr
    def testSVM(self,clfArr):
        correct = 0
        setSize = len(self.targets)
        with torch.no_grad():
            predicted = np.zeros(shape=(self.classifiersNum, setSize))
            top2lst = list()
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                for i in range(self.classifiersNum):


                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs,pool, featureLayer,_ = self.net(inputs)

                    # predicted[i, batch_idx * self.batch:(batch_idx + 1) * self.batch] += clfArr[i].score_samples(featureLayer.cpu().numpy())
                    if self.takeFeat:

                        predicted[i,batch_idx*self.batch:(batch_idx+1)*self.batch] += clfArr[i].predict(featureLayer.cpu().numpy())
                    else:
                        predicted[i, batch_idx * self.batch:(batch_idx + 1) * self.batch] += clfArr[i].predict(outputs.cpu().numpy())
                progress_bar(i, self.classifiersNum)
                top2, _ = torch.topk(m(outputs), 2)
                top2 = top2.cpu().numpy()
                top1 = top2[:,0].tolist()
                top2 = top2[:, 1].tolist()
                for ind in range(len(top1)):
                    top2lst.append(top1[ind] - top2[ind])
            predictedTot = list()
            unknown = 0
            tooknown = 0
            detected = list()
            unknownlst = list()
            tooknownlst = list()
            for j in range(setSize):

                decision = False
                # predictedTot.append(int(np.argmax(predicted[:,j])/(1+self.overclass)))
                # continue
                for i in range(0,self.classifiersNum,1 + self.overclass):
                    if predicted[i][j] == 1 or predicted[i + self.overclass][j] ==1:
                        if decision == True:
                            predictedTot[j] = -1
                            tooknown +=1
                            break
                        else:
                            decision = True
                            if self.overclass:
                                predictedTot.append(int(i/2))
                            else:
                                predictedTot.append(i)
                if decision == False:
                    predictedTot.append(-2)
                    unknown +=1
                print(j)

                if predictedTot[j] == -1:
                    tooknownlst.append(top2lst[j])
                elif predictedTot[j] == -2:
                    unknownlst.append(top2lst[j])
                else:
                    detected.append(top2lst[j])
            predictedTot = torch.tensor(predictedTot, device=self.device)
            targetsFull = torch.tensor(self.targets, device=self.device)
            correct += predictedTot.eq(targetsFull).sum().item()
        print("too known is",tooknown)
        print("too unknown is", unknown)
        print("acc is ",100*correct/setSize)
        x=3