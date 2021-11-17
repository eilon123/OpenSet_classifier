
import numpy as np
import torch
from tsne import colors_per_class,showtsne
from createDataset import createDataset
from sklearn.manifold import TSNE
import copy
from numpy import linalg as LA
from statistics import variance


def getCentroids(net,device,trainset,classes):
    global best_acc
    net.eval()

    with torch.no_grad():
        for ind ,label in enumerate(classes):
            trainsetFiltered = copy.deepcopy(trainset)
            trainloader,_ = createDataset(trainsetFiltered, np.array([label]),False,batchSize=500)
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs,_, featureLayer,innerLayers= net(inputs)


                featureLayer = featureLayer.cpu().numpy()
                layer1 = innerLayers[0].cpu().numpy()
                layer2 = innerLayers[1].cpu().numpy()
                layer3 = innerLayers[2].cpu().numpy()
                layer4 = innerLayers[3].cpu().numpy()

                if batch_idx == 0:
                    features = featureLayer
                    layer1Tot = layer1
                    layer2Tot = layer2
                    layer3Tot = layer3
                    layer4Tot = layer4
                else:
                    features = np.concatenate((features, featureLayer))
                    # layer1Tot = np.concatenate((layer1Tot, layer1))
                    # layer2Tot = np.concatenate((layer2Tot, layer2))
                    # layer3Tot = np.concatenate((layer3Tot, layer3))
                    # layer4Tot = np.concatenate((layer4Tot, layer4))

            if ind == 0:
                centroidsFeat = ([features.mean(axis=0)])
                clusterVar = ([features.var(axis=0)])

                # centroidsL1 = ([layer1Tot.mean(axis=0)])
                # centroidsL2 = ([layer2Tot.mean(axis=0)])
                # centroidsL3 = ([layer3Tot.mean(axis=0)])
                # centroidsL4 = ([layer4Tot.mean(axis=0)])

            else:
                centroidsFeat = np.concatenate((centroidsFeat, ([features.mean(axis=0)])),axis=0)
                clusterVar = np.concatenate((clusterVar, ([features.var(axis=0)])))
                # centroidsL1 = np.concatenate((centroidsL1, ([layer1Tot.mean(axis=0)])), axis=0)
                # centroidsL2 = np.concatenate((centroidsL2, ([layer2Tot.mean(axis=0)])), axis=0)
                # centroidsL3 = np.concatenate((centroidsL3, ([layer3Tot.mean(axis=0)])), axis=0)
                # centroidsL4 = np.concatenate((centroidsL4, ([layer4Tot.mean(axis=0)])), axis=0)


        return centroidsFeat,clusterVar#centroidsL1,centroidsL2,centroidsL3,centroidsL4

def classify(net,device,testloader,centroidsFeat,centroidsL1,centroidsL2,centroidsL3,centroidsL4):
    net.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0

    total = 0
    hist = np.zeros(len(testloader.dataset))
    lastChange = np.zeros(len(testloader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,extra,featureLayer,layer1,layer2,layer3,layer4 = net(inputs)

            predictedFL = np.array([])
            predictedl1 = np.array([])
            predictedl2 = np.array([])
            predictedl3 = np.array([])
            predictedl4 = np.array([])

            print(batch_idx)
            for ind,row in enumerate(layer1.cpu().numpy()):
                dist = np.array([])
                for i in range(len(centroidsFeat)):
                    dist = np.append(dist, (LA.norm(row - centroidsL1[i])))
                predictedl1 = np.append(predictedl1, np.argmin(dist))
                lastChange[ind] = predictedl1[ind]
            for ind,row in enumerate(layer2.cpu().numpy()):
                dist = np.array([])
                for i in range(len(centroidsFeat)):
                    dist = np.append(dist, (LA.norm(row - centroidsL2[i])))
                predictedl2 = np.append(predictedl2, np.argmin(dist))
                if lastChange[ind] != predictedl2[ind]:
                    hist[int(batch_idx * testloader.batch_size + ind)] += 1
                lastChange[ind] = predictedl2[ind]

            for ind,row in enumerate(layer3.cpu().numpy()):
                dist = np.array([])
                for i in range(len(centroidsFeat)):
                    dist = np.append(dist, (LA.norm(row - centroidsL3[i])))
                predictedl3 = np.append(predictedl3, np.argmin(dist))
                if lastChange[ind] != predictedl3[ind]:
                    hist[int(batch_idx * testloader.batch_size + ind)] += 1
                lastChange[ind] = predictedl3[ind]
            for ind,row in enumerate(layer4.cpu().numpy()):
                dist = np.array([])
                for i in range(len(centroidsFeat)):
                    dist = np.append(dist, (LA.norm(row - centroidsL4[i])))
                predictedl4 = np.append(predictedl4,np.argmin(dist))
                if lastChange[ind] != predictedl4[ind]:
                    hist[int(batch_idx * testloader.batch_size + ind)] += 1
                lastChange[ind] = predictedl4[ind]
            for ind,row in enumerate(featureLayer.cpu().numpy()):
                predictedFL = np.append(predictedFL,np.argmin((LA.norm (row - centroidsFeat, ord=2, axis=1))))
                if lastChange[ind] != predictedFL[ind]:
                    hist[int(batch_idx * testloader.batch_size + ind)] += 1
            predictedFL[predictedFL == 2] = 8
            predictedFL[predictedFL == 3] = 9

            predictedl1[predictedl1 == 2] = 8
            predictedl1[predictedl1 == 3] = 9

            predictedl2[predictedl2 == 2] = 8
            predictedl2[predictedl2 == 3] = 9

            predictedl3[predictedl3 == 2] = 8
            predictedl3[predictedl3 == 3] = 9

            predictedl4[predictedl4 == 2] = 8
            predictedl4[predictedl4 == 3] = 9

            total += targets.size(0)

            predictedFL = torch.tensor(predictedFL, device=device)
            correct1 += predictedFL.eq(targets).sum().item()

            predictedl1 = torch.tensor(predictedl1, device=device)
            correct2 += predictedl1.eq(targets).sum().item()


            predictedl2 = torch.tensor(predictedl2, device=device)
            correct3 += predictedl2.eq(targets).sum().item()


            predictedl3 = torch.tensor(predictedl3, device=device)
            correct4 += predictedl3.eq(targets).sum().item()

            predictedl4 = torch.tensor(predictedl4, device=device)
            correct5 += predictedl4.eq(targets).sum().item()

    torch.histc(hist)
    print("by FL: ", correct1 / total)
    print("by L1: ", correct2 / total)
    print("by L2: ", correct3 / total)
    print("by L3: ", correct4 / total)
    print("by L4: ", correct5 / total)


def getCentroidsTSNE(net,device,trainset,classes):
    global best_acc
    net.eval()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False, num_workers=0)
    trainsetFiltered = trainset

    with torch.no_grad():


        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs ,featureLayer = net(inputs)

            current_outputs = outputs.cpu().numpy()
            if batch_idx == 0:
                output = current_outputs
            else:
                output = np.concatenate((output, current_outputs))

        tsne = TSNE(n_components=2).fit_transform(output)

        TSNEcenteroids = calcCentroids(tsne,trainset,classes)



        return tsne,TSNEcenteroids


def calcCentroids(tsne,trainset,classes):
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    firstTime = True
    for ind, _ in enumerate(colors_per_class):
        if ind not in classes:
            continue
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(trainset.targets) if l == ind]

        # extract the coordinates of the points of this class only
        x = np.mean(np.take(tx, indices))
        y = np.mean(np.take(ty, indices))
        if firstTime == True:
            TSNEcenteroids = ([np.array([x, y])])
            firstTime = False
        else:
            TSNEcenteroids = np.concatenate((TSNEcenteroids, ([np.array([x, y])])), axis=0)

    return TSNEcenteroids

def classifyTSNE(tsne,testloader,TSNEcenteroids,testIdx,device):

    test_loss = 0
    correct = 0
    total = 0
    hist = np.zeros(shape=(10, 10))

    correctTot = 0
    correctUnknownTot = 0
    unknownTot = 0
    numClasses = 10
    tx = tsne[testIdx:, 0]
    ty = tsne[testIdx:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    predicted = np.array([])
    targetTot = np.array([])
    for t1,t2 in zip(tx,ty):



        predicted = np.append(predicted, np.argmin((LA.norm(np.array([t1 ,t2]) - TSNEcenteroids, ord=2,axis=1))))

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        targetTot = np.append(targetTot,targets.cpu().numpy())
        total += targets.size(0)
        predicted = torch.tensor(predicted, device=device)
        correct += predicted[testloader.batch_size * batch_idx : testloader.batch_size * (batch_idx+1)].eq(targets).sum().item()

    # showtsne(tsne,targetTot)
    return correct /total


def gethistproduct(net,device,loader,centroids):
    net.eval()

    hist = np.zeros(len(loader.dataset))
    dotarray = [ [[] for l in range(10)] for i in range(10)]
    meanArr = [[[] for l in range(10)] for i in range(10)]
    stdArr = [[[] for l in range(10)] for i in range(10)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, featureLayer,_, _, _, _ = net(inputs)
            dotres = list()
            for i in range(10):
                FL = featureLayer.cpu().numpy()
                norm2 = np.transpose(np.sqrt(np.sum(FL ** 2, axis=1)))
                dotres.append(np.dot(FL/norm2.reshape(500,1), centroids[i]/np.sqrt(np.sum(centroids[i]**2))))
            for i,tar in enumerate(targets):
                for j ,res in enumerate(dotres):
                    dotarray[tar][j].append(dotres[j][i])
    for i in range(0, 100):
        # Find row and column index
        row = i // 10
        col = i % 10

        meanArr[row][col] = np.mean(dotarray[row][col])
        stdArr[row][col] = np.std(dotarray[row][col])

    return meanArr,stdArr
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
