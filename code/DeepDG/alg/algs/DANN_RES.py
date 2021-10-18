# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm

# concate fusion
class DANN_RES_C(Algorithm):

    def __init__(self, args,DANN):
        """

        :param args: hyper-parameter:args.bottleneck_a
        :param DANN: pretrained
        """
        super(DANN_RES_C, self).__init__(args)

        self.featurizer_a = get_fea(args)
        self.bottleneck_a = common_network.feat_bottleneck(
            self.featurizer_a.in_features, args.bottleneck_a, args.layer)
        self.featurizer_b = DANN.featurizer
        self.bottleneck_b = DANN.bottleneck
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck_a+args.bottleneck, args.classifier)
        self.args = args

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        print('y.shape',all_y.shape)
        print('y',all_y)
        all_z_a = self.bottleneck_a(self.featurizer_a(all_x))
        all_z_b = self.bottleneck_b(self.featurizer_b(all_x)).detach()
        all_z = torch.cat((all_z_a,all_z_b),dim=1)
        print("z.shape,z_a.shape,z_b.shape")
        print(all_z.shape,all_z_a.shape,all_z_b.shape)
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss = classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item()}

    def predict(self, x):
        all_z_a = self.bottleneck_a(self.featurizer_a(x))
        all_z_b = self.bottleneck_b(self.featurizer_b(x))
        all_z = torch.cat((all_z_a,all_z_b),dim=1)
        all_preds = self.classifier(all_z)
        return all_preds


class DANN_RES_A(Algorithm):

    def __init__(self, args,DANN):
        """

        :param args: hyper-parameter: fusion rate mu
        :param DANN: pretrained
        """
        super(DANN_RES_A, self).__init__(args)

        self.featurizer_a = get_fea(args)
        self.bottleneck_a = common_network.feat_bottleneck(
            self.featurizer_a.in_features, args.bottleneck, args.layer)
        self.featurizer_b = DANN.featurizer
        self.bottleneck_b = DANN.bottleneck
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        self.args = args

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        print('y.shape',all_y.shape)
        print('y',all_y)
        all_z_a = self.bottleneck_a(self.featurizer_a(all_x))
        all_z_b = self.bottleneck_b(self.featurizer_b(all_x)).detach()
        # feature fusion
        all_z = all_z_b+self.args.mu*all_z_a
        print("z.shape,z_a.shape,z_b.shape")
        print(all_z.shape,all_z_a.shape,all_z_b.shape)
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss = classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class': classifier_loss.item()}

    def predict(self, x):
        all_z_a = self.bottleneck_a(self.featurizer_a(x))
        all_z_b = self.bottleneck_b(self.featurizer_b(x))
        all_z = all_z_b + self.args.mu * all_z_a
        all_preds = self.classifier(all_z)
        return all_preds