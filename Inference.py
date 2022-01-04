import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from stats import *
from utils import progress_bar
from tsne import *
from createDataset import *
import wandb

criterion = nn.CrossEntropyLoss()
criterion3 = nn.MSELoss()
criterion_unsoftmax = nn.NLLLoss()
m = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)
ent = 0


class Inference:
    def __init__(self, args, device, net, optimizer, trainloader, testloader, targets, classes, reduce=0,
                 trainloader2=0,
                 ent=0,best_acc=0 ):
        super(Inference, self).__init__()

        self.trans = args.trans
        self.overclass = args.overclass
        self.extraLayer = args.extraLayer
        self.openset = args.openset
        self.featperclass = args.featperclass
        self.featTsne = args.f
        self.union = args.union
        self.directTrans = args.directTrans

        self.trainloader = trainloader
        self.trainloader2nd = trainloader2
        self.trainBsize = trainloader.batch_size
        self.testloader = testloader
        self.testBsize = testloader.batch_size
        self.tstSize = len(testloader) * testloader.batch_size
        self.optimizer = optimizer
        self.net = net
        self.device = device

        self.targets = targets

        self.ent = ent
        self.extraclass = args.extraclass
        self.tsne = args.tsne
        self.epochTSNE = args.epochTSNE

        self.address = createFolders(args)
        self.transActive = False
        self.class_names = classes
        self.pool = args.pool
        self.reduce = reduce
        self.mnist = args.mnist
        self.unsave = args.unsave
        self.best_acc = best_acc
    def setTrans(self):
        self.overclass = False
        for idx, l in enumerate(next(self.net.children()).children()):
            for param in l.parameters():
                param.requires_grad = idx > 5

                if idx > 5:
                    if param.dim() > 1:  # wieghts
                        torch.nn.init.kaiming_uniform_(param)
                        # torch.nn.init.xavier_uniform_(param)
                    else:  # Bias
                        torch.nn.init.zeros_(param)
        self.transActive = True
        self.optimizer.param_groups[0]["lr"] = 1e-2
    def setWeights(self,highest,lowest):
        hist = np.zeros(512)

        self.highest = torch.unique(torch.cat(highest,0))
        self.lowest = torch.unique(torch.cat(lowest,0))
        print(len(self.highest))
        for i in range(len(highest)):
            for j in range(10):
                hist[highest[i][j].item()] += 1
        uniqueFeathigh = torch.cat(highest,0)
        # for feat,i in enumerate
        # for idx, l in enumerate(next(self.net.children()).children()):
        #     for param in l.parameters():
        #
        #
        #         if idx > 5:
        #             if param.dim() > 1:  # wieghts
        #                 for i in range(np.shape(param)[0]):
        #                     for j in range(np.shape(param)[1]):
        #                         if j in highest[i]:
        #                             nn.init.constant_(param[i][j], 1)
        #
        #                         else:
        #                             torch.nn.init.zeros_(param[i][j])


    def setPH2(self):
        for idx, l in enumerate(next(self.net.children()).children()):
            for param in l.parameters():
                param.requires_grad = idx > 5

                if idx > 5:
                    if param.dim() > 1:  # wieghts
                        torch.nn.init.kaiming_uniform_(param)
                        # torch.nn.init.xavier_uniform_(param)
                    else:  # Bias
                        torch.nn.init.zeros_(param)

    def train(self, epoch, finalTest):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        start = time.time()
        train_lossCE = 0
        train_lossUnif = 0
        train_lossUniq = 0
        all_preds = torch.Tensor([]).cuda()
        all_tar = torch.Tensor([]).cuda()
        wandb.watch(self.net)

        if self.transActive:
            trainloader = self.trainloader2nd
        else:
            trainloader = self.trainloader

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs, pool, featureLayer, _ = self.net(inputs)
            current_outputs = outputs.detach().cpu().numpy()
            loss = 0
            if self.overclass:
                uniquenessLoss, uniformLoss, newOutputs = self.ent.CalcEntropyLoss(outputs, self.transActive)

                if self.pool:
                    lossCE = criterion(pool, targets)
                else:
                    lossCE = criterion_unsoftmax(m(newOutputs), targets)

                loss += lossCE
                if not self.transActive:
                    loss += uniformLoss
                    loss += uniquenessLoss
                    # wandb.log({"uniq loss per batch (train)": (uniquenessLoss.item() / self.trainBsize)})
                    # wandb.log({"unif loss per batch (train)": (uniformLoss.item() / self.trainBsize)})
                _, predictedext = outputs.max(1)

                # Stats
                train_lossCE += lossCE.item()
                train_lossUnif += uniformLoss.item()
                train_lossUniq += uniquenessLoss.item()

                # Not Stats
                _, predicted = newOutputs.max(1)
                if batch_idx == 0:
                    output = current_outputs
                    totalPredict = predictedext.cpu()
                else:
                    output = np.concatenate((output, current_outputs))
                    totalPredict = np.concatenate((totalPredict, predictedext.cpu()))
            else:
                loss = criterion(outputs, targets)

                # Not Stats
                _, predicted = outputs.max(1)
                if batch_idx == 0:
                    output = current_outputs

                else:
                    output = np.concatenate((output, current_outputs))

            # print('\t',loss.item())
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            all_preds = torch.cat((all_preds, predicted), dim=0)
            all_tar = torch.cat((all_tar, targets), dim=0)

        if self.tsne or finalTest or epoch == self.epochTSNE:
            if self.overclass and not (self.union):
                showtsne(output, totalPredict, self.address[0:13 + 4 * self.mnist], "train", numClasses=20)
            else:
                showtsne(output, self.targets[:], self.address[0:13 + 4 * self.mnist], "train")
        wandb.log({"train loss": (train_loss / (batch_idx + 1))})

        if self.overclass:
            wandb.log({"uniq loss (train)": (train_lossUniq / (batch_idx + 1))})
            wandb.log({"unif loss (train)": (train_lossUnif / (batch_idx + 1))})
            wandb.log({"CE loss (train)": (train_lossCE / (batch_idx + 1))})

        # self.reduce.step()
        print("LR:", self.optimizer.param_groups[0]["lr"])
        end = time.time()

        print("time is ", abs(int(start - end)), " sec")

    def test(self, epoch, finalTest):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        featurelist = torch.empty(size=(0, 512)).cuda()
        gradeHist = -np.ones(shape=(self.tstSize, 10 * (int(self.overclass) - int(self.union * self.overclass>0) + 1)))
        gradeHist = torch.from_numpy(gradeHist)
        gradeHist = gradeHist.cuda()
        histIdx = np.zeros(10)
        test_lossCE = 0
        test_lossUnif = 0
        test_lossUniq = 0
        totalPredict = 0
        output = 0
        predictedext = 0
        all_preds = torch.Tensor([]).cuda()
        all_tar = torch.Tensor([]).cuda()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, pool, featureLayer, _ = self.net(inputs)
                featHigh = copy.deepcopy(featureLayer)

                for i in range(512):
                    if i in self.highest:
                        continue
                    else:
                        for feat in featHigh:
                            feat[i] = 0
                outputs1 = self.net.module.quickForward(featHigh)
                for i in range(512):
                    if i in self.lowest:
                        continue
                    else:
                        for feat in featureLayer:
                            feat[i] = 0
                outputs2 = self.net.module.quickForward(featureLayer)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                # outputs = softmax(outputs)
                prob, predicted = outputs.max(1)

                # _, predicted1 = outputs1.max(1)
                # _, predicted2 = outputs2.max(1)
                #
                # for i,pred in enumerate(predicted):
                #     if predicted1[i] == predicted2[i]:
                #         predicted[i] = predicted1[i]
                #     else:
                #         predicted[i] = 9
                # predicted[prob<0.97] = 9
                if self.overclass:
                    uniquenessLoss, uniformLoss, newOutputs = self.ent.CalcEntropyLoss(outputs, self.transActive)
                    if not (self.openset):
                        lossCE = criterion_unsoftmax(m(newOutputs), targets)
                        loss += lossCE
                    loss += uniformLoss
                    loss += uniquenessLoss

                    _, predictedext = outputs.max(1)

                    newOutputs = newOutputs.cuda()
                    _, predicted = newOutputs.max(1)
                if (epoch == self.epochTSNE or finalTest or self.tsne) and self.overclass:
                    x=3
                    gradeHist = calcHist(outputs, predicted, histIdx, gradeHist, self.extraclass)
                if self.openset:
                    targets[targets>4] = 9
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                test_loss = loss.item()
                if self.overclass:
                    if not (self.openset):
                        test_lossCE += lossCE.item()
                    test_lossUnif += uniformLoss
                    test_lossUniq += uniquenessLoss

                if finalTest or epoch == self.epochTSNE:
                    featurelist = torch.cat((featurelist, featureLayer), dim=0)

                totalPredict, output = getFeat(batch_idx, self.overclass, predicted, predictedext, totalPredict,
                                               outputs, output)

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (loss / (batch_idx + 1), 100. * correct / total, correct, total))
                # Save checkpoint.
                all_preds = torch.cat((all_preds, predicted), dim=0)
                all_tar = torch.cat((all_tar, targets), dim=0)

        acc = 100. * correct / total
        wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=all_tar.cpu().numpy(),
                                                           preds=all_preds.cpu().numpy(),
                                                           class_names=self.class_names)})
        if ((acc > self.best_acc or finalTest) and not (self.directTrans)) and not self.unsave:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            torch.save(state, self.address)
            self.best_acc = acc
            if finalTest:
                calcStats(featurelist, self.address[0:13 + 4 * self.mnist], self.net)
        if self.tsne or finalTest or epoch == self.epochTSNE:
            if self.featperclass or self.featTsne:
                features = featurelist.cpu().numpy()
            else:
                features = output

            for j in range(len(np.transpose(features))):
                fig, axs = plt.subplots(10 + 10*self.overclass, 1)
                fig.set_size_inches(18.5, 10.5, forward=True)
                if j == 10:
                    break
                # for i in range(len(np.transpose(features))):
                #
                #     x = features[all_tar.cpu().numpy()==j].flatten()
                #     # scoresperClass = features[all_tar.cpu().numpy()==j][i]
                #     scoresperClass = x[i::10+10*self.overclass]
                #     axs[i].hist(scoresperClass, bins=200,range=(np.min(features[all_tar.cpu().numpy()==j])-0.5,np.max(features[all_tar.cpu().numpy()==j])+0.5))


            # plt.show()
                s = self.address[0:11]+ 'hist target '+ str(j)
                # axs.set_title(s)
                plt.savefig(s)

            showHist(gradeHist, self.address[0:13 + 4 * self.mnist], len(outputs[0]), self.union)

            # plt.figure()
            # plt.hist(totalPredict, bins=100,range=())
            # plt.savefig(self.address[0:13] + "predict_hist")
            if self.overclass and not (self.union) and not (self.openset):
                showtsne(features, totalPredict, self.address[0:13 + 4 * self.mnist], "test", numClasses=20)
            elif self.overclass and self.openset:
                showtsne(features, totalPredict, self.address[0:13 + 4 * self.mnist], "test")
            else:
                showtsne(features, self.targets[:], self.address[0:13 + 4 * self.mnist], "test")

        wandb.log({"test loss": (test_loss / (batch_idx + 1))})
        if self.overclass:
            wandb.log({"uniq loss (test)": (test_lossUniq / (batch_idx + 1))})
            wandb.log({"unif loss (test)": (test_lossUnif / (batch_idx + 1))})
            # wandb.log({"CE loss (test)": (test_lossCE / (batch_idx + 1))})
        return acc
