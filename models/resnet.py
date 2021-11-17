'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, extraClasses=1, extraLayer=1, extraFeat=False, pool=False,
                 d=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.extraClasses = extraClasses
        self.num_classes = num_classes
        self.extraLayer = extraLayer
        self.extraFeat = extraFeat
        self.pool = pool

        # Base Network
        self.conv1 = nn.Conv2d(d, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Flavours
        if self.extraFeat:  # Split before the last layer
            self.class_fc = []
            self.extraFC = nn.Linear(512 * block.expansion, 128 * num_classes)
            self.sumOfAllFears = nn.Linear(num_classes * 640 * block.expansion, 1 * num_classes * extraClasses)
            for i in range(num_classes):
                self.class_fc.append(nn.Linear(640 * block.expansion, extraClasses).to('cuda'))

        elif self.extraClasses == 1 or self.extraLayer == 1:  # Regular Train
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        else:  # Class Overloading
            self.class_fc = []
            self.extraFC = nn.Linear(512 * block.expansion, self.extraClasses * num_classes)
            for i in range(num_classes):
                self.class_fc.append(nn.Linear(self.extraClasses * block.expansion, 1).to('cuda'))
            # for i in range(num_classes):
            #     self.nfc[i] = nn.Linear(extraClasses * block.expansion,1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        innerlayers = list()
        innerlayers.append(layer1)
        innerlayers.append(layer2)
        innerlayers.append(layer3)
        innerlayers.append(layer4)

        x = F.avg_pool2d(layer4, 4)
        x = x.view(x.size(0), -1)
        featureLayer = x
        pool = 0
        if self.extraClasses == 1:
            x = self.linear(x)
            if self.pool:
                pool = self.scores(x)

        if self.extraFeat:
            featEx = self.extraFC(x)

            out_extention = []
            out_extention1 = []
            for i in range(0, self.num_classes):
                featPerclass = featEx[:, 128 * i:128 * (i + 1)]
                out_extention1.append(self.class_fc[i](torch.cat((featPerclass, x), dim=1)))
            #     out_extention.append(torch.cat((featPerclass , out),dim=1))
            # finalFeat = self.sumOfAllFears(out_extention)
            #
            # for i in range(0, self.num_classes):
            #     pre = finalFeat[:, 640 * i * self.extraClasses:640 * (i+1) * self.extraClasses]
            #
            #     out_extention1.append(self.class_fc[i](pre))
            out = torch.stack(out_extention1, axis=0).squeeze().T
            out = torch.stack(out_extention1, axis=2).squeeze().T
            out = torch.flatten(out, start_dim=0, end_dim=1).T
            return out, featEx, featureLayer, innerlayers

        if self.extraClasses != 1 and not self.extraLayer:
            x = self.extraFC(x)
            extraC = x
            out_extention = []
            for i in range(0, self.num_classes):
                pre = x[:, i * self.extraClasses:i * self.extraClasses + self.extraClasses]
                out_extention.append(self.class_fc[i](pre))
            out = torch.stack(out_extention, axis=0).squeeze().T

            return out, pool, featureLayer, None

        return x, pool, featureLayer, innerlayers


def ResNet18(num_classes, extraClasses=1, extraLayer=1, extraFeat=False, pool=False, d=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, extraClasses, extraLayer, extraFeat=extraFeat, pool=pool, d=d)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
# net.requires_grad_(False)
# list(net.children())[0].linear.requires_grad_(True)
