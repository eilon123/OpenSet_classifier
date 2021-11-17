import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):

        if x.ndim == 1:
            # p = x / sum(x)
            # b = (p * torch.log(p)/torch.log(2))
            if torch.any(x != 0):
                b = F.softmax(x, dim=0) * F.log_softmax(x, dim=0)
            else:
                b = torch.zeros(1).cuda()
            # b = torch.max((x != 0)) * F.softmax(x, dim=0) * F.log_softmax(x, dim=0)
        else:
            exist = torch.zeros(len(x))

            # for i in range(extraClasses):

            exist[:] = torch.any((x != 0), 1)

            exist = exist.cuda()

            b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
            for i in range(len(b)):
                if exist[i] == 0:
                    b[i, 0] = 0

                    # b = exist * F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -b.sum()
        return b


class EntropyLoss():
    def __init__(self, args, device):
        super(EntropyLoss, self).__init__()
        self.isTrans = args.trans
        self.extraclass = args.extraclass
        self.device = device
        self.batch_size = args.batch
        self.criterion = HLoss()
        self.Kunif = args.Kunif
        self.Kuniq = args.Kuniq

    def CalcEntropyLoss(self, outputs, transActive):
        newOutputs = torch.zeros(len(outputs), int(np.shape(outputs)[1] / (self.extraclass))).cuda()
        if not (transActive):
            mask = torch.zeros(len(outputs), int(np.shape(outputs)[1]))
            _, predicted = outputs.max(1)
        scores = F.softmax(outputs, dim=1)
        if not (transActive):
            for row in range(len(outputs)):

                mask[row,int(predicted[row].item()/5)*self.extraclass:(int(predicted[row].item()/5)+1)*self.extraclass] +=1


            mask = mask.to(self.device)
            sumProb = (1 / self.batch_size) * torch.sum(mask * outputs, dim=0)
        uniformLoss = 0
        uniquenessLoss = 0
        maxEnt = 0
        unifarr = torch.ones( int(np.shape(outputs)[1]))
        unifarr = unifarr.to(self.device)
        k = 0
        for i in range(np.shape(newOutputs)[1]):
            if not (transActive):
                uniformLoss += self.Kunif * (self.criterion(sumProb[k:k + self.extraclass]))
                uniquenessLoss += (self.Kuniq / self.batch_size) * self.criterion(
                    mask[:, k:k + self.extraclass] * outputs[:, k:k + self.extraclass])
                maxEnt -= self.Kunif*(self.criterion(unifarr[k:k+self.extraclass]))
            newOutputs[:, i] = torch.sum(scores[:, k: k + self.extraclass], dim=1)
            k += self.extraclass
            if self.isTrans and not (transActive) and i == 4:
                break
        # print(uniformLoss.item())
        print(uniformLoss*100/maxEnt)
        return uniquenessLoss, uniformLoss, newOutputs
