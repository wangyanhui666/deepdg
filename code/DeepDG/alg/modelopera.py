# coding=utf-8
import time

import torch
from network import img_network


def get_fea(args):
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    elif args.net.startswith('res'):
        net = img_network.ResBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
    return net



def accuracy(network, loader):
    correct = 0
    total = 0
    time1=0
    time2=0
    time3=0
    sss=time.time()
    network.eval()
    with torch.no_grad():
        for data in loader:

            sss0=time.time()
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            sss1=time.time()

            time1+=sss1-sss0
            p = network.predict(x)
            sss2=time.time()

            time2+=sss2-sss1
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
            sss3=time.time()
            time3+=sss3-sss2
    end=time.time()
    network.train()

    print('data to cuda time={}'.format(time1))
    print('predict time {}'.format(time2))
    print('caculate label time {}'.format(time3))
    print('total eval time {}'.format(end-sss))
    return correct / total
