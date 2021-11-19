<<<<<<< HEAD
import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import sys
import time
import numpy as np
import argparse
import copy
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ,alg_class_dict
from datautil.getdataloader import get_img_dataloader
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int,
                        default=256, help='bottleneck dim')
    parser.add_argument('--bottleneck_a', type=int,
                        default=256, help='top path bottleneck dim')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='office')
    parser.add_argument('--data_dir', type=str, default='', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"], help='bottleneck normalization style')
    parser.add_argument('--logdir',type=str,default=None,help='tensorboard logdir path')
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--mu',type=float,default=1,help='DANN_RES_A feature fusion rate')
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shownet',action='store_true')
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--visual_data',type=str,default='train',help='visualization data')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.step_per_epoch = 100000000000
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args

def select_n_random(fea, labels,img, n=1000):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(fea) == len(labels)

    perm = torch.randperm(len(fea))
    return fea[perm][:n], labels[perm][:n],img[perm][:n]

class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels

    def plot_tsne(self, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                     color=plt.cm.Set1(self.labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.show()

def get_features(network,loader):
    flag=0
    with torch.no_grad():
        for data in loader:

            x=data[0].cuda().float()
            y=data[1].long().numpy()
            fea=network.feature(x).cpu().numpy()
            if flag==0:
                fea_arr=fea
                clabel_arr=y
                img_arr=x.cpu()
                flag=1
            else:
                fea_arr=np.concatenate((fea_arr,fea),axis=0)
                clabel_arr=np.concatenate((clabel_arr,y))
                img_arr=torch.cat((img_arr,x.cpu()),0)
            print(fea.shape,img_arr.shape)
    return fea_arr,clabel_arr,img_arr

if __name__ == '__main__':
    args = get_args()
    print('=======hyper-parameter used========')
    args.checkpoint='../../model/model.pkl'
    s = print_args(args, [])
    print(s)
    writer = SummaryWriter('./output/visual/target/')
    train_loaders, eval_loaders = get_img_dataloader(args)
    acc_type_list = ['target']
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    classes=alg_class_dict(args)
    print('use pretrained DANN')
    check_point=torch.load(args.checkpoint)
    algorithm = algorithm_class(args)
    algorithm.load_state_dict(check_point['model_dict'])
    algorithm.cuda()
    algorithm.eval()
    flag=0
    for item in acc_type_list:
        print(item)
        for i in eval_name_dict[item]:
            fea_arr,clabel_arr,img_tenosr=get_features(algorithm,eval_loaders[i])
            if flag==0:
                flag=1
                fea_arr_full=fea_arr
                clabel_arr_full=clabel_arr
                img_tenosr_full=img_tenosr
            else:
                fea_arr_full=np.concatenate((fea_arr_full,fea_arr),axis=0)
                clabel_arr_full=np.concatenate((clabel_arr_full,clabel_arr))
                img_tenosr_full=torch.cat((img_tenosr_full,img_tenosr),0)
    print(fea_arr_full.shape)
    print(clabel_arr_full.shape)
    print(img_tenosr_full.shape)

    clabel_list=[classes[lab] for lab in clabel_arr_full]
    print(clabel_list)
    fea_arr_full,clabel_arr_full,img_tenosr_full=select_n_random(fea_arr_full,clabel_arr_full,img_tenosr_full,n=1000)
    print(fea_arr_full.shape)
    print(clabel_arr_full.shape)
    print(img_tenosr_full.shape)
    writer.add_embedding(fea_arr_full,
                         metadata=clabel_arr_full,
                         label_img=img_tenosr_full)
=======
import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import sys
import time
import numpy as np
import argparse
import copy
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ,alg_class_dict
from datautil.getdataloader import get_img_dataloader
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int,
                        default=256, help='bottleneck dim')
    parser.add_argument('--bottleneck_a', type=int,
                        default=256, help='top path bottleneck dim')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='office')
    parser.add_argument('--data_dir', type=str, default='', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"], help='bottleneck normalization style')
    parser.add_argument('--logdir',type=str,default=None,help='tensorboard logdir path')
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--mu',type=float,default=1,help='DANN_RES_A feature fusion rate')
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shownet',action='store_true')
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--visual_data',type=str,default='train',help='visualization data')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.step_per_epoch = 100000000000
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args

def select_n_random(fea, labels,img, n=1000):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(fea) == len(labels)

    perm = torch.randperm(len(fea))
    return fea[perm][:n], labels[perm][:n],img[perm][:n]

class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels

    def plot_tsne(self, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                     color=plt.cm.Set1(self.labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.show()

def get_features(network,loader):
    flag=0
    with torch.no_grad():
        for data in loader:

            x=data[0].cuda().float()
            y=data[1].long().numpy()
            fea=network.feature(x).cpu().numpy()
            if flag==0:
                fea_arr=fea
                clabel_arr=y
                img_arr=x.cpu()
                flag=1
            else:
                fea_arr=np.concatenate((fea_arr,fea),axis=0)
                clabel_arr=np.concatenate((clabel_arr,y))
                img_arr=torch.cat((img_arr,x.cpu()),0)
            print(fea.shape,img_arr.shape)
    return fea_arr,clabel_arr,img_arr

if __name__ == '__main__':
    args = get_args()
    print('=======hyper-parameter used========')
    args.checkpoint='../../model/model.pkl'
    s = print_args(args, [])
    print(s)
    writer = SummaryWriter('./output/visual/target/')
    train_loaders, eval_loaders = get_img_dataloader(args)
    acc_type_list = ['target']
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    classes=alg_class_dict(args)
    print('use pretrained DANN')
    check_point=torch.load(args.checkpoint)
    algorithm = algorithm_class(args)
    algorithm.load_state_dict(check_point['model_dict'])
    algorithm.cuda()
    algorithm.eval()
    flag=0
    for item in acc_type_list:
        print(item)
        for i in eval_name_dict[item]:
            fea_arr,clabel_arr,img_tenosr=get_features(algorithm,eval_loaders[i])
            if flag==0:
                flag=1
                fea_arr_full=fea_arr
                clabel_arr_full=clabel_arr
                img_tenosr_full=img_tenosr
            else:
                fea_arr_full=np.concatenate((fea_arr_full,fea_arr),axis=0)
                clabel_arr_full=np.concatenate((clabel_arr_full,clabel_arr))
                img_tenosr_full=torch.cat((img_tenosr_full,img_tenosr),0)
    print(fea_arr_full.shape)
    print(clabel_arr_full.shape)
    print(img_tenosr_full.shape)

    clabel_list=[classes[lab] for lab in clabel_arr_full]
    print(clabel_list)
    fea_arr_full,clabel_arr_full,img_tenosr_full=select_n_random(fea_arr_full,clabel_arr_full,img_tenosr_full,n=1000)
    print(fea_arr_full.shape)
    print(clabel_arr_full.shape)
    print(img_tenosr_full.shape)
    writer.add_embedding(fea_arr_full,
                         metadata=clabel_arr_full,
                         label_img=img_tenosr_full)
>>>>>>> f0a7744 (add DAAN first model)
