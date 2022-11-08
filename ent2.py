import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
kl_loss = nn.KLDivLoss(reduction="batchmean")
sft = nn.LogSoftmax(dim=1)

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        if x.nelement() ==0:
            return  0

        elif x.dim() == 1:
            if torch.sum(torch.abs(x)) == 0:
                return 0
            b = F.softmax(x,dim=0) * F.log_softmax(x,dim=0)
            b = -1.0 * b.mean()
        else:
            b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
            b = -1.0 * b.mean()
        return b
class Entropy2():
    def __init__(self, args, device,Nclasses=6):
        super(Entropy2, self).__init__()
        self.isTrans = args.trans
        self.extraclass = args.extraclass
        self.device = device
        self.batch_size = args.batch
        self.criterion = HLoss()
        self.Kunif = args.Kunif
        self.Kuniq = args.Kuniq
        self.stopindex = args.stop
        self.Nclasses = Nclasses
    def CalcEntropyLoss(self, outputs):
        newOutputs = torch.zeros(len(outputs), np.shape(outputs)[1] // self.extraclass).cuda()
        if not self.isTrans:
            mask = torch.zeros(len(outputs), int(np.shape(outputs)[1]))
            _, predicted = outputs.max(1)

            for row in range(len(outputs)):
                st = (predicted[row].item() // self.extraclass) * self.extraclass
                ed = st + self.extraclass
                mask[row, st:ed] += 1

            mask = mask.to(self.device)
            sumProb = (1 / self.batch_size) * torch.sum(mask * outputs, dim=0)
        scores = F.softmax(outputs, dim=1)

        uniformLoss = 0
        uniquenessLoss = 0
        unifarr = torch.ones(int(np.shape(outputs)[1]))
        unifarr = unifarr.to(self.device)
        k = 0
        maxEnt = (self.criterion(unifarr[0 * self.extraclass:
                    (0 + 1) * self.extraclass]))
        self.Nclasses=2
        for i in range(self.Nclasses):
            if not self.isTrans:


                uniformLoss -= self.Kunif * self.criterion(sumProb[i * self.extraclass:(i + 1) * self.extraclass])
                # uniformLoss += self.Kunif * (maxEnt - self.criterion(sumProb[i * self.extraclass:(i + 1) * self.extraclass]))
                # input = mask[:, i * self.extraclass:(i + 1) * self.extraclass] * outputs[:, k:k + self.extraclass]
                input = mask[:, i * self.extraclass:(i + 1) * self.extraclass] * outputs[:,i * self.extraclass:(i + 1) * self.extraclass]
                input = input[torch.abs(input.sum(dim=1)) != 0]
                uniquenessLoss += (self.Kuniq ) * self.criterion(input)



            newOutputs[:, i] = torch.sum(
                scores[:, i * self.extraclass:
                          (i + 1) * self.extraclass], dim=1)

            if self.isTrans and i == 4:
                break
        print(2*maxEnt + uniformLoss/self.Kunif)
        return uniquenessLoss, uniformLoss, newOutputs
    def calcent(self,outputs):
        return self.criterion(outputs)
    def CalcEntropyLoss_perclass(self, outputs):

        if not self.isTrans:

            sumProb = (1 / self.batch_size) * torch.sum(outputs, dim=0)
        scores = F.softmax(outputs, dim=1)

        uniformLoss = 0
        uniquenessLoss = 0
        unifarr = torch.ones(int(np.shape(outputs)[1]))
        unifarr = unifarr.to(self.device)
        k = 0

        for i in range(np.shape(outputs)[1]):
            if not self.isTrans:


                uniformLoss -= self.Kunif * self.criterion(sumProb)

                input = outputs
                input = input[torch.abs(input.sum(dim=1)) != 0]
                uniquenessLoss += (self.Kuniq ) * self.criterion(input)




        return uniquenessLoss, uniformLoss
    def CalcKLloss(self, outputs,scores):
        loss = 0
        prob = sft(scores)
        for i in range(len(outputs)):
            loss += kl_loss(sft(outputs[i]),prob)
            if i == self.stopindex:
                break
        print(loss)
        return loss