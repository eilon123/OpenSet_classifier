import time

from stats import *
from utils import progress_bar
from tsne import *
from createDataset import *
import wandb

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
criterion3 = nn.MSELoss()
criterion_unsoftmax = nn.NLLLoss()
criterion1 = nn.L1Loss()
criterion1Tst = nn.L1Loss(reduction='none')
criterion3Tst = nn.MSELoss(reduction='none')

sft = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)
ent = 0


class Inference:
    def __init__(self, args, device, net, optimizer, trainloader, testloader, targets, classes, reduce=0,
                 ent=0,best_acc=0 ):
        super(Inference, self).__init__()

        self.trans = args.trans
        self.overclass = args.overclass
        self.extraLayer = args.extraLayer
        self.openset = args.openset
        self.featperclass = args.featperclass
        self.featTsne = args.f
        self.union = args.union
        self.stopindex = args.stop
        self.trainloader = trainloader
        self.trainBsize = trainloader.batch_size
        self.testloader = testloader
        self.testBsize = testloader.batch_size
        self.tstSize = len(testloader) * testloader.batch_size
        self.optimizer = optimizer
        self.net = net
        self.device = device

        self.targets = targets

        self.ent = ent
        self.kl = args.kl
        self.extraclass = args.extraclass
        self.tsne = args.tsne
        self.epochTSNE = args.epochTSNE

        self.address = createFolders(args)

        self.class_names = classes
        self.pool = args.pool
        self.reduce = reduce
        self.mnist = args.mnist
        self.unsave = args.unsave or len(args.load_path) > 0
        self.best_acc = best_acc
        self.AE = BasicAE().cuda()
        self.Nclasses = (10 * int(args.openset==0) + 6*int(args.openset)) * (1+(args.overclass*args.extraclass-1))
        if self.trans:
            self.Nclasses = 10
        if args.ae:
            self.optimizer = optim.Adam(self.AE.parameters(), lr=args.lr,
                                   weight_decay=5e-4)

        self.constLayer = args.constLayer
        self.cluster = args.cluster
        self.meanVec = None
        self.arcface = args.arcface
        self.ismeanVecReady = False

        self.deepclassifier = args.deepclassifier
        if self.deepclassifier:
            self.quicknet = Quicknet_ctor(num_classes=10)
            self.deepoptimizer = optim.SGD(self.quicknet.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4 )
            if args.resume and args.train ==0:
                address = "quicknet/checkpoint/ckptquicky.pth"
                checkpoint = torch.load(address)
                new_weights = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
                self.quicknet.load_state_dict(new_weights)


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
        self.optimizer.param_groups[0]["lr"] = 1e-2
    def setM(self,m):
        self.m = m
    def setWeights(self,highest,lowest,order):
        hist = np.zeros(512)

        self.highest = torch.unique(torch.cat(highest,0))
        self.lowest = torch.unique(torch.cat(lowest,0))
        print(len(self.highest))
        for i in range(len(highest)):
            for j in range(10):
                hist[highest[i][j].item()] += 1
        uniqueFeathigh = torch.cat(highest,0)
        self.order = order


    def setclasses(self,selected):
        self.selectedClasses = selected
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
    def setMeanVec(self,is_test):
        if self.ismeanVecReady and is_test:
            return
        self.meanVec = torch.zeros(10, 512).cuda()
        for idx, l in enumerate(next(self.net.children()).children()):
            for param in l.parameters():

                if idx > 5:
                    if param.dim() > 1:  # wieghts
                        for i in range(10):
                            self.meanVec[i] = param[i]
        self.ismeanVecReady = True
    def normalizeW(self):

        w = torch.zeros(10, 512).cuda()
        w_norm = torch.zeros(10,1).cuda()
        for idx, l in enumerate(next(self.net.children()).children()):
            for param in l.parameters():

                if idx > 5:
                    if param.dim() > 1:  # wieghts
                        for i in range(10):
                            w_norm[i] = torch.norm(param[i])
                            w[i] = torch.nn.functional.normalize(param[i],dim=0)

        return w,w_norm
    def Getquicknet(self):
        if self.deepclassifier:
            return self.quicknet
        return 0
    def train(self, epoch, finalTest):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        correctDeep = torch.zeros(4)
        total = 0
        start = time.time()
        scoreslist = torch.empty(size=(0, 10)).cuda()
        train_lossCE = 0
        train_lossUnif = 0
        train_lossUniq = 0
        train_lossConst = 0

        all_preds = torch.Tensor([]).cuda()
        all_tar = torch.Tensor([]).cuda()
        wandb.watch(self.net)



        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.deepclassifier:
                with torch.no_grad():
                    outputs, featureLayer, innerlayers = self.net(inputs)
                outputlist = self.quicknet(innerlayers)
            else:
                outputs, featureLayer, deepLayers = self.net(inputs)
            current_outputs = outputs.detach().cpu().numpy()
            loss = 0
            lossCluster = 0
            if self.deepclassifier:
                predicteddeep = list()
                for ind in range(len(outputlist)):
                    loss += criterion(outputlist[ind], targets)

                    _, predicted = outputlist[ind].max(1)
                    predicteddeep.append(predicted)

            elif self.overclass:
                self.ent.setEpoch(epoch)
                uniquenessLoss, uniformLoss, newOutputs = self.ent.CalcEntropyLoss(outputs)
                lossCE = criterion_unsoftmax((newOutputs), targets)
                loss += lossCE
                _, predicted = newOutputs.max(1)
                if self.kl :
                    for i in range(len(deepLayers)):

                        kl_loss ,u,_= self.ent.CalcEntropyLoss(deepLayers[i])
                        loss += kl_loss + u
                        if i == self.stopindex:
                            break
                    # kl_loss = 0.1*self.ent.CalcKLloss(deepLayers ,outputs)

                if not self.trans and epoch > 1 or True:
                    loss += uniformLoss
                    loss += uniquenessLoss
                    # print(loss)
                    # print(uniquenessLoss.item())
                    # wandb.log({"uniq loss per batch (train)": (uniquenessLoss.item() / self.trainBsize)})
                    # wandb.log({"unif loss per batch (train)": (uniformLoss.item() / self.trainBsize)})
                _, predictedext = outputs.max(1)

                # Statsddd
                train_lossCE += lossCE.item()

                train_lossUnif += uniformLoss.item()
                train_lossUniq += uniquenessLoss.item()

                # Not Stats

                if batch_idx == 0:
                    output = current_outputs
                    totalPredict = predictedext.cpu()
                else:
                    output = np.concatenate((output, current_outputs))
                    totalPredict = np.concatenate((totalPredict, predictedext.cpu()))


            else:

                if epoch > 5 and self.cluster:
                    self.setMeanVec(self, is_test=0)


                    for i in range(len(targets)):
                        lossCluster += (1/self.trainBsize) * criterion3(featureLayer[i],self.meanVec[targets[i]])

                elif self.arcface:


                    s = torch.norm(featureLayer,dim=1)

                    featureNorm = torch.nn.functional.normalize(featureLayer)
                    featureNorm = torch.transpose(featureNorm, 0, 1)
                    w,Wnorm = self.normalizeW()
                    Wnorm = torch.transpose(Wnorm, 0, 1)
                    scores = w @ featureNorm
                    one_hot = torch.nn.functional.one_hot(targets, 10)
                    m = self.m * one_hot
                    theta = torch.arccos(scores)
                    marginaltargetlogit = s*torch.cos(theta + torch.transpose(m, 0, 1))
                    outputs = torch.transpose(marginaltargetlogit, 0, 1)


                loss += criterion(outputs, targets)
                if self.cluster:
                    loss += lossCluster
                # Not Stats
                _, predicted = outputs.max(1)
                if batch_idx == 0:
                    output = current_outputs

                else:
                    output = np.concatenate((output, current_outputs))
            if self.constLayer:
                sumofScores = torch.sum(outputs, axis=1)
                ones = torch.ones(self.trainBsize).cuda()
                factor = 0.001
                lossConst =factor*criterion3(sumofScores,ones)
                loss += lossConst
                train_lossConst += lossConst.item() / factor
            # print('\t',loss.item())
            loss.backward()
            if self.deepclassifier:
                self.deepoptimizer.step()
            else:
                self.optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if self.deepclassifier:
                for ind in range(len(outputlist)):
                    correctDeep[ind] += predicteddeep[ind].eq(targets).sum().item()

            if self.deepclassifier:
                progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc layer 0 : %.3f%% | Acc layer 1 : %.3f%%  | Acc layer 2 : %.3f%%  | Acc layer 3 : %.3f%% | Acc final  : %.3f%% '
                             % (train_loss / (batch_idx + 1), 100. * correctDeep[0] / total,100. * correctDeep[1] / total,100. * correctDeep[2] / total,100. * correctDeep[3] / total,100. * correct/ total))
            else:
                progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            all_preds = torch.cat((all_preds, predicted), dim=0)
            all_tar = torch.cat((all_tar, targets), dim=0)

        # if self.tsne or finalTest or epoch == self.epochTSNE:
        #     if self.overclass and not (self.union):
        #         showtsne(output, totalPredict, self.address, "train", numClasses=20)
        #     else:
        #         showtsne(output, self.targets[:], self.address, "train")
        wandb.log({"train loss": (train_loss / (batch_idx + 1))})
        wandb.log({"train cluster loss": (lossCluster / (batch_idx + 1))})
        # mean0, var0 = get_score_table(scoreslist, all_tar, self.address,all_preds)
        # wandb.log({"train mean": mean0[0]})
        # wandb.log({"train std": var0[0]})
        if self.overclass:
            wandb.log({"uniq loss (train)": (train_lossUniq / (batch_idx + 1))})
            wandb.log({"unif loss (train)": (train_lossUnif / (batch_idx + 1))})
            wandb.log({"CE loss (train)": (train_lossCE / (batch_idx + 1))})
        if self.constLayer:
            wandb.log({"const layer loss (train)": (train_lossConst / (batch_idx + 1))})
        # self.reduce.step()
        print("LR:", self.optimizer.param_groups[0]["lr"])
        end = time.time()

        print("time is ", abs(int(start - end)), " sec")
        return 0,(train_loss / (batch_idx + 1))

    def test(self, epoch, finalTest,inspect = False):

        saveModel = finalTest
        finalTest = finalTest or inspect
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        correctDeep = torch.zeros(4)
        featurelist = torch.empty(size=(0, 512)).cuda()
        featurelist_2 = torch.empty(size=(0, 65536)).cuda()
        featurelist_3 = torch.empty(size=(0, 32768)).cuda()
        featurelist_4 = torch.empty(size=(0, 16384)).cuda()
        featurelist_5 = torch.empty(size=(0, 8192)).cuda()
        scoreslist = torch.empty(size=(0, 10*(1+self.overclass))).cuda()
        gradeHist = -np.ones(shape=(self.tstSize, self.Nclasses))
        gradeHist = torch.from_numpy(gradeHist)
        gradeHist = gradeHist.cuda()
        histIdx = np.zeros(10)
        test_lossCE = 0
        test_lossUnif = 0
        test_lossUniq = 0
        test_lossConst = 0
        lossCluster = 0
        totalPredict = 0
        output = 0
        predictedext = 0
        all_preds = torch.Tensor([]).cuda()
        all_preds_over = torch.Tensor([]).cuda()
        all_tar = torch.Tensor([]).cuda()
        pin = []

        pout = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, featureLayer, deeplayer = self.net(inputs)
                if self.deepclassifier:
                    outputlist = self.quicknet(deeplayer)

                featHigh = copy.deepcopy(featureLayer)

                # for i in range(512):
                #     if i in self.highest:
                #         continue
                #     else:
                #         for feat in featHigh:
                #             feat[i] = 0
                # outputs1 = self.net.module.quickForward(featHigh)
                # for i in range(512):
                #     if i in self.lowest:
                #         continue
                #     else:
                #         for feat in featureLayer:
                #             feat[i] = 0
                # outputs2 = self.net.module.quickForward(featureLayer)

                loss = criterion(outputs, targets)

                if self.constLayer:
                    sumofScores = torch.sum(outputs, axis=1)
                    ones = torch.ones(self.trainBsize).cuda()
                    lossConst = 0.1 * criterion3(sumofScores, ones)
                    loss += lossConst
                    test_lossConst += lossConst.item()
                test_loss += loss.item()
                # outputs = softmax(outputs)
                prob, predicted = outputs.max(1)

                if self.deepclassifier:
                    predicteddeep = list()
                    for ind in range(len(outputlist)):
                        loss += criterion(outputlist[ind], targets)

                        _, predicted = outputlist[ind].max(1)
                        predicteddeep.append(predicted)

                if self.overclass:
                    self.ent.setEpoch(epoch)
                    uniquenessLoss, uniformLoss, newOutputs = self.ent.CalcEntropyLoss(outputs)
                    if not (self.openset):
                        lossCE = criterion_unsoftmax(sft(newOutputs), targets)
                        loss += lossCE
                    loss += uniformLoss
                    loss += uniquenessLoss

                    _, predictedext = outputs.max(1)

                    newOutputs = newOutputs.cuda()
                    _, predicted = newOutputs.max(1)

                if epoch > 5 and self.cluster:
                    self.setMeanVec(is_test=True)
                    for i in range(len(targets)):
                        lossCluster += (1 / self.testBsize) * criterion3(featureLayer[i], self.meanVec[targets[i]])
                if (epoch == self.epochTSNE or finalTest or self.tsne) and self.overclass :
                    gradeHist = calcHist(outputs, predicted, histIdx, gradeHist, self.extraclass)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if self.deepclassifier:
                    for ind in range(len(outputlist)):
                        correctDeep[ind] += predicteddeep[ind].eq(targets).sum().item()


                test_loss = loss.item()
                if self.overclass:
                    if not (self.openset):
                        test_lossCE += lossCE.item()
                    test_lossUnif += uniformLoss
                    test_lossUniq += uniquenessLoss

                if finalTest or epoch == self.epochTSNE or True:

                    # featurelist_2 = torch.cat((featurelist_2, deeplayer[0].flatten(start_dim=1)), dim=0)

                    # featurelist_3 = torch.cat((featurelist_3, deeplayer[1].flatten(start_dim=1)), dim=0)
                    # featurelist_4 = torch.cat((featurelist_4, deeplayer[2].flatten(start_dim=1)), dim=0)
                    # featurelist_5 = torch.cat((featurelist_5, deeplayer[3].flatten(start_dim=1)), dim=0)
                    featurelist = torch.cat((featurelist, featureLayer), dim=0)
                    # scoreslist = torch.cat((scoreslist, outputs), dim=0)

                totalPredict, output = getFeat(batch_idx, self.overclass, predicted, predictedext, totalPredict,
                                               outputs, output)

                if self.deepclassifier:
                    progress_bar(batch_idx, len(self.testloader),
                                 'Loss: %.3f | Acc layer 0 : %.3f%% | Acc layer 1 : %.3f%%  | Acc layer 2 : %.3f%%  | Acc layer 3 : %.3f%% | Acc final  : %.3f%% '
                                 % (loss / (batch_idx + 1), 100. * correctDeep[0] / total,
                                    100. * correctDeep[1] / total, 100. * correctDeep[2] / total,
                                    100. * correctDeep[3] / total, 100. * correct / total))
                else:
                    progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (loss / (batch_idx + 1), 100. * correct / total, correct, total))
                # Save checkpoint.
                all_preds = torch.cat((all_preds, predicted), dim=0)
                if self.overclass:
                    all_preds_over = torch.cat((all_preds_over, predictedext), dim=0)
                all_tar = torch.cat((all_tar, targets), dim=0)
                # if batch_idx == 17 and True:
                #     break



        acc = 100. * correct / total
        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=all_tar.cpu().numpy(),
        #                                                    preds=all_preds.cpu().numpy(),
        #                                                    class_names=self.class_names)})
        # if (acc > self.best_acc or saveModel) and not (self.directTrans) and not self.unsave:
        if (acc > self.best_acc ) and not (self.trans) and not self.unsave:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            torch.save(state, self.address + '/checkpoint/ckpt.pth')
            if self.deepclassifier:
                print("another foking network")
                state = {
                    'net': self.quicknet.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                s = self.address + "quicky"
                torch.save(state, s)

            self.best_acc = acc
        if finalTest:
            x=4
            # calcStats(featurelist, self.address[0:13 + 4 * self.mnist], self.net)
            # calcinFeatCorr(featurelist, all_tar,self.order)
            # calcFeatCorr(featurelist, self.order, all_tar,)
            # calcFeatCorr(featurelist_2, self.order, all_tar,2 )
            # calcFeatCorr(featurelist_3, self.order, all_tar,3 )
            # calcFeatCorr(featurelist_4, self.order, all_tar, 4)
            # calcFeatCorr(featurelist_5, self.order, all_tar, 5)
        if (self.tsne or finalTest or epoch == self.epochTSNE)  and True :
            if self.featperclass or self.featTsne:
                features = featurelist.cpu().numpy()
            else:
                features = output


            if self.overclass:
                showHist(gradeHist, self.address , len(outputs[0]),self.extraclass, self.union,epoch)

            # plt.figure()
            # plt.hist(totalPredict, bins=100,range=())
            # plt.savefig(self.address[0:13] + "predict_hist")
            tsne_name = str(epoch)
            if finalTest and not (inspect):
                tsne_name = 'final'
            if self.overclass and not (self.union) and not (self.openset):
                showtsne(features, totalPredict, self.address, "test", numClasses=20,epoch=tsne_name)
            elif self.overclass and self.openset:
                showtsne(features, totalPredict, self.address, "test",epoch=tsne_name)
            else:
                showtsne(features, self.targets[:], self.address, "test",epoch=tsne_name)
        L2 = 0.0005*get_L2_loss(self.net)
        wandb.log({"L2 loss": (L2.item())})
        wandb.log({"test loss": (test_loss / (batch_idx + 1))})
        wandb.log({"test cluster loss": (lossCluster / (batch_idx + 1))})

        if self.overclass:
            wandb.log({"uniq loss (test)": (test_lossUniq / (batch_idx + 1))})
            wandb.log({"unif loss (test)": (test_lossUnif / (batch_idx + 1))})
            wandb.log({"uniq normalized": (10*test_lossUniq / (batch_idx + 1))})
            wandb.log({"unif normalized": (0.5*test_lossUnif / (batch_idx + 1))})
            # wandb.log({"CE loss (test)": (test_lossCE / (batch_idx + 1))})
        if self.constLayer:
            wandb.log({"const layer loss (test)": (test_lossConst / (batch_idx + 1))})
        return acc , test_loss / (batch_idx + 1),0


    def AEtrain(self,epoch):
        train_loss = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            loss = 0
            loss1=0
            loss2=0
            self.optimizer.zero_grad()
            # tmp = torch.ones((1,512,4,4))
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs , _,featureLayer,innerLayer = self.net(inputs)
            bott = innerLayer[3]
            auto= self.AE(bott)
            loss = criterion1(inputs, auto)


            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | total is %d) '
                         % (train_loss / (batch_idx + 1),  total))

    def AEtest(self,epoch,finalTest):

        test_loss = 0
        total = 0
        print("test")
        hist = np.zeros(shape=(10,1000))
        histIdx = np.zeros(10)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                loss = 0
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, _, _, innerLayer = self.net(inputs)
                bott = innerLayer[3]
                auto = self.AE(bott)

                if finalTest:
                    loss1 = criterion1Tst(inputs,auto)

                    for i in range(len(targets)):
                        tar = targets[i].item()
                        idx = int(histIdx[tar])
                        hist[tar,idx] = torch.var(loss1[i].flatten()).item()
                        # hist[tar,idx] = torch.sqrt(torch.var(auto[i]))
                        # print(torch.mean(loss1[i].flatten()).item())
                        # out = torchvision.utils.make_grid(auto[i])
                        # f, axarr = plt.subplots(1, 2)
                        # f.suptitle(str(targets[i]))
                        # img = inputs[i].cpu().numpy()
                        # img = (img / 2 + 0.5)
                        #
                        # img = np.transpose(img, (1, 2, 0))
                        # axarr[ 0].imshow(img)
                        # img = auto[i] / 1.0
                        # axarr[1].imshow(img.T.cpu().numpy())
                        # # plt.imshow(out.T.cpu().numpy())
                        # plt.show()
                        histIdx[tar] += 1
                    x=3

                loss += criterion1(inputs, auto)



                test_loss += loss.item()
                total += targets.size(0)

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | total is %d) '
                             % (test_loss / (batch_idx + 1),  total))
        if test_loss < self.best_acc and not self.unsave:
            self.best_acc = test_loss
            print('Saving..')
            state = {
                'net': self.AE.state_dict(),
                'acc': loss,
                'epoch': epoch,
            }

            torch.save(state, self.address + '/checkpoint/ckpt.pth')
        if finalTest:
            fig, axs = plt.subplots(10 + 10 * self.overclass, 1)
            fig.set_size_inches(18.5, 10.5, forward=True)

            for i in range(10):

                axs[i].hist(hist[i,:], bins=200,range=(0,3))
            plt.show()
        return test_loss



