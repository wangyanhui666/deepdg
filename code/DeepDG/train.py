# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse
import copy
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, alg_class_dict, Tee, img_param_init, print_environ
from datautil.getdataloader import get_img_dataloader
from torch.utils.tensorboard import SummaryWriter
from feature_vis import get_features,select_n_random
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

    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--local_run',action='store_true')
    parser.add_argument('--logdir', type=str, default=None, help='tensorboard logdir path')

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
    parser.add_argument('--tokennum',type=int,default=64,help='number of common embedding token')
    parser.add_argument('--visual',action='store_true')
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


if __name__ == '__main__':

    args = get_args()
    if not args.local_run:
        import tensorflow as tf
        import tensorboard as tb
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    set_random_seed(args.seed)
    print('=======hyper-parameter used========')
    s = print_args(args, [])
    print(s)
    writer = SummaryWriter(args.logdir)
    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    if args.algorithm=='DANN_RES_C' or args.algorithm=='DANN_RES_A':
        print('use pretrained DANN')
        DANN=alg.get_algorithm_class('DANN')(args)
        check_point=torch.load(args.checkpoint)
        DANN.load_state_dict(check_point['model_dict'])
        algorithm = algorithm_class(args,DANN).cuda()
    else:
        algorithm = algorithm_class(args).cuda()
    algorithm.train()

    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)

    acc_record = {}
    loss_record=np.zeros(len(loss_list))
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    sss = time.time()
    time1=0
    time2=0
    for epoch in range(args.max_epoch):
        for iter_num in range(args.step_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            # sss2=time.time()
            # time1+=sss2-sss1
            step_vals = algorithm.update(minibatches_device, opt, sch)
            for i ,item in enumerate(loss_list):
                loss_record[i]+=step_vals[item]
        loss_record=loss_record/args.step_per_epoch

        for i, item in enumerate(loss_list):
            writer.add_scalar('loss/{}'.format(item), loss_record[i], epoch)
            s += (item + '_loss:%.4f,' % loss_record[i])
        print(s[:-1])
        loss_record=np.zeros(len(loss_list))

        print('read data time{}'.format(time1))
        print('update time {}'.format(time2))

        print('training cost time: %.4f' % (time.time() - sss))
        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''

            for item in acc_type_list:
                print(item)
                acc_record[item] = np.mean(np.array([modelopera.accuracy(
                    algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                s += (item+'_acc:%.4f,' % acc_record[item])
                writer.add_scalar('acc/{}'.format(item), acc_record[item], epoch)
            print(s[:-1])
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
                best_algorithm = copy.deepcopy(algorithm)
                best_epoch = epoch
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time()-sss))
            algorithm_dict = algorithm.state_dict()

    save_checkpoint('best_model.pkl',best_algorithm,args)
    print('Best model saved!')
    save_checkpoint('model.pkl', algorithm, args)
    writer.add_scalar('result acc',target_acc,global_step=best_epoch)


    print('DG result: %.4f' % target_acc)
    if args.shownet==True:
        all_x = torch.cat([data[0].float() for data in minibatches_device])
        writer.add_graph(algorithm,all_x)
    if args.visual:
        print('add embedding in tensorboard')
        algorithm.cuda()
        algorithm.eval()

        classes = alg_class_dict(args)


        for item in acc_type_list:
            print(item)
            for n,i in enumerate(eval_name_dict[item]):
                fea_arr,clabel_arr,img_tenosr=get_features(algorithm,eval_loaders[i])
                if n==0:
                    fea_arr_full=fea_arr
                    clabel_arr_full=clabel_arr
                    img_tenosr_full=img_tenosr
                else:
                    fea_arr_full=np.concatenate((fea_arr_full,fea_arr),axis=0)
                    clabel_arr_full=np.concatenate((clabel_arr_full,clabel_arr))
                    img_tenosr_full=torch.cat((img_tenosr_full,img_tenosr),0)

            clabel_list = [classes[lab] for lab in clabel_arr_full]
            fea_arr_full, clabel_arr_full, img_tenosr_full = select_n_random(fea_arr_full, clabel_arr_full, img_tenosr_full,n=1000)
            writer.add_embedding(fea_arr_full,
                             metadata=clabel_arr_full,
                             label_img=img_tenosr_full,
                             tag=item)

    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('target acc:%.4f' % (target_acc))
