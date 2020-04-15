from torchvision import models
import torch
from torch import nn
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet


class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.T = nn.Parameter(torch.ones(1, ))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return x * self.T

    def loss(self, x, t):
        return self.ce(x, t)


class MyResnet(nn.Module):
    def __init__(self, net='resnet101', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyResnet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'resnet18':
            resnet = models.resnet18(pretrained)
            bl_exp = 1
        elif net == 'resnet34':
            resnet = models.resnet34(pretrained)
            bl_exp = 1
        elif net == 'resnet50':
            resnet = models.resnet50(pretrained)
            bl_exp = 4
        elif net == 'resnet101':
            resnet = models.resnet101(pretrained)
            bl_exp = 4
        elif net == 'resnet152':
            resnet = models.resnet152(pretrained)
            bl_exp = 4
        else:
            raise Warning("Wrong Net Name!!")
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AvgPool2d(int(size / 32), stride=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)
        self.last_fc = nn.Linear(512 * bl_exp, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        x = self.last_fc(x)
        return x


class MyEfficientnet(nn.Module):
    def __init__(self, net='efficientnetb0', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyEfficientnet, self).__init__()
        self.dropout_flag = dropout_flag
        if net[:13] == 'efficientnetb' and len(net) == 14 and int(net[-1]) <= 7:
            if pretrained:
                efficientnet = EfficientNet.from_pretrained('efficientnet-b' + net[-1], num_classes=num_classes)
            else:
                efficientnet = EfficientNet.from_name('efficientnet-b' + net[-1], num_classes=num_classes)

        else:
            raise Warning("Wrong Net Name!!")

        self.efficientnet = efficientnet

    def forward(self, x):
        x = self.efficientnet(x)
        return x


class MySeResnext(nn.Module):
    def __init__(self, net='seresnext50', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MySeResnext, self).__init__()
        self.dropout_flag = dropout_flag
        self.size = size

        if net == 'seresnext50':
            seresnext50 = ptcv_get_model("seresnext50_32x4d", pretrained=pretrained)
            if self.size == 224:
                self.model = nn.Sequential(*(list(seresnext50.children())[0]))
            else:
                self.model = nn.Sequential(*(list(seresnext50.children())[0][:-1]))
        elif net == 'seresnext101':
            seresnext101 = ptcv_get_model("seresnext101_32x4d", pretrained=pretrained)
            if self.size == 224:
                self.model = nn.Sequential(*(list(seresnext101.children())[0]))
            else:
                self.model = nn.Sequential(*(list(seresnext101.children())[0][:-1]))
        else:
            raise Warning("Wrong Net Name!!")

        if self.size != 224:
            self.avgpool = nn.AvgPool2d(int(size / 32), stride=1)

        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)

        self.last_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)

        if self.size != 224:
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)

        x = self.last_fc(x)
        return x


class MyDensenet(nn.Module):
    def __init__(self, net='densenet169', pretrained=False, num_classes=1, dropout_flag=False, size=512):
        super(MyDensenet, self).__init__()
        self.dropout_flag = dropout_flag
        if net == 'densenet121':
            densenet = models.densenet121(pretrained)
            num_features = 1024
        elif net == 'densenet169':
            densenet = models.densenet169(pretrained)
            num_features = 1664
        elif net == 'densenet201':
            densenet = models.densenet201(pretrained)
            num_features = 1920
        elif net == 'densenet161':
            densenet = models.densenet161(pretrained)
            num_features = 2208
        else:
            raise Warning("Wrong Net Name!!")
        self.densenet = nn.Sequential(*(list(densenet.children())[0]))
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=int(size / 32), stride=1)
        if self.dropout_flag:
            self.dropout = nn.Dropout(0.5)
        self.last_fc = nn.Linear(num_features, num_classes)

    def forward(self, x, w_code=False):
        x = self.densenet(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout_flag:
            x = self.dropout(x)
        out = self.last_fc(x)
        if w_code:
            return x, out
        return out


def get_model(net, pretrained, num_classes, dropout, size):
    if net[:8] == 'densenet':
        return MyDensenet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif net[:6] == 'resnet':
        return MyResnet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif "seresnext" in net:
        return MySeResnext(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout, size=size).to(
            'cuda')
    elif "efficientnet" in net:
        return MyEfficientnet(net=net, pretrained=pretrained, num_classes=num_classes, dropout_flag=dropout,
                              size=size).to('cuda')
    else:
        raise Warning("Wrong Net Name!!")


def get_criterion(lossname, dataset_classes=[[0], [1], [2], [3], [4], [5], [6], [7]]):
    # Defining the loss
    whole_training_stats = [0.173232994, 0.522600954, 0.126555506, 0.032807043, 0.104471005, 0.008066499, 0.008755103,
                            0.023510895]
    training_stats = []
    for c in dataset_classes:
        training_stats.append(np.sum([whole_training_stats[c_i] for c_i in c]))
    weights = np.divide(1, training_stats, dtype='float32')

    criterion = None
    if lossname == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device='cuda'))
    else:
        print("WRONG LOSS NAME")
    return criterion


def get_optimizer(n, learning_rate, optname='SGD'):
    optimizer = None
    if optname == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, n.parameters()), lr=learning_rate)
    elif optname == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, n.parameters()), lr=learning_rate)
    else:
        print("WRONG OPTIMIZER NAME")

    return optimizer


def get_scheduler(optimizer, schedname=None):
    scheduler = None
    if schedname == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, threshold=0.004)

    return scheduler
