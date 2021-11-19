# coding=utf-8
import copy
from typing import Optional, Any, Union, Callable
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from torchvision import models


res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101": models.resnext101_32x8d}





class DAAN(Algorithm):
    def __init__(self,args):
        super(DAAN, self).__init__(args)

        self.featurizer=get_fea(args)
        #b,512
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        # self.conv1=nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,bias=False)
        #b,256
        self.common_embedding=nn.Parameter(torch.randn(1,args.tokennum,256))
        self.cross_attn_layer=CrossAttentionEncoderLayer(d_model=256,nhead=8,batch_first=True)
        self.classifier_o = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        self.classifier_n = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        self.discriminator_o = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.discriminator_n = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.args=args

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_z = self.bottleneck(self.featurizer(all_x))
        #b,256
        disc_input_origin = all_z
        disc_input_origin = Adver_network.notReverseLayerF.apply(disc_input_origin, self.args.alpha)
        disc_out_origin = self.discriminator_o(disc_input_origin)
        disc_labels = torch.cat([
            torch.full((data[0].shape[0], ), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])

        disc_loss_origin = F.cross_entropy(disc_out_origin, disc_labels)
        all_preds_origin = self.classifier_o(all_z)
        classifier_loss_origin = F.cross_entropy(all_preds_origin, all_y)
        loss_origin = classifier_loss_origin+disc_loss_origin

        x = all_z.view(all_z.size(0), 1, -1)
        #b,1,256

        common_embedding_batch=self.common_embedding.repeat(x.size(0),1,1)

        x=self.cross_attn_layer(x,common_embedding_batch)
        #b,1,256
        x=x.view(x.size(0),-1)
        #b,256

        disc_input_new = x
        disc_input_new = Adver_network.ReverseLayerF.apply(disc_input_new, self.args.alpha)
        disc_out_new=self.discriminator_n(disc_input_new)
        disc_loss_new=F.cross_entropy(disc_out_new,disc_labels)
        all_preds_new=self.classifier_n(x)
        classifier_loss_new=F.cross_entropy(all_preds_new,all_y)
        loss_new=classifier_loss_new+disc_loss_new
        loss=loss_new+loss_origin
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class_o': classifier_loss_origin.item(),'class_n': classifier_loss_new.item(), 'dis_o': disc_loss_origin.item(),'dis_n': disc_loss_new.item()}

    def predict(self, x):
        z=self.bottleneck(self.featurizer(x))
        x=z.view(z.size(0), 1, -1)
        common_embedding_batch = self.common_embedding.repeat(x.size(0),1,1)
        x = self.cross_attn_layer(x, common_embedding_batch)
        x = x.view(x.size(0),-1)
        return self.classifier_n(x)

    def feature(self,x):
        z = self.bottleneck(self.featurizer(x))
        x = z.clone().view(z.size(0),1,-1)
        common_embedding_batch = self.common_embedding.repeat(x.size(0),1,1)
        x = self.cross_attn_layer(x, common_embedding_batch)
        x = x.view(x.size(0),-1)
        return x




class DAAN_first(Algorithm):
    def __init__(self,args):
        super(DAAN_first, self).__init__(args)

        self.featurizer=get_fea(args)
        #b,512
        self.avgpool=nn.AdaptiveAvgPool2d((16,16))
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.bottleneck_n = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        # self.conv1=nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,bias=False)
        #b,256
        self.common_embedding=nn.Parameter(torch.randn(1,args.tokennum,256))
        self.cross_attn_layer=CrossAttentionEncoderLayer(d_model=256,nhead=8,batch_first=True)
        self.classifier_o = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        self.classifier_n = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        self.discriminator_o = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.discriminator_n = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.domain_num - len(args.test_envs))
        self.args=args

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_z = self.bottleneck(self.featurizer(all_x))
        #b,256
        disc_input_origin = all_z
        disc_input_origin = Adver_network.notReverseLayerF.apply(disc_input_origin, self.args.alpha)
        disc_out_origin = self.discriminator_o(disc_input_origin)
        disc_labels = torch.cat([
            torch.full((data[0].shape[0], ), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])

        disc_loss_origin = F.cross_entropy(disc_out_origin, disc_labels)
        all_preds_origin = self.classifier_o(all_z)
        classifier_loss_origin = F.cross_entropy(all_preds_origin, all_y)
        loss_origin = classifier_loss_origin+disc_loss_origin


        x=self.featurizer.first_layer(all_x)
        #b,64,56,56
        x=self.avgpool(x)
        #b,64,16,16
        x=x.view(x.size(0),x.size(1),-1)
        #b,64,256
        common_embedding_batch=self.common_embedding.repeat(x.size(0),1,1)

        x=self.cross_attn_layer(x,common_embedding_batch)
        #b,64,256
        x=x.view(x.size(0),x.size(1),16,16)
        #b,64,16,16
        x=self.featurizer.last_four_layers(x)
        #b,512,1,1

        x=x.view(x.size(0),-1)
        x=self.bottleneck_n(x)
        #b,256
        disc_input_new = x
        disc_input_new = Adver_network.ReverseLayerF.apply(disc_input_new, self.args.alpha)
        disc_out_new=self.discriminator_n(disc_input_new)
        disc_loss_new=F.cross_entropy(disc_out_new,disc_labels)
        all_preds_new=self.classifier_n(x)
        classifier_loss_new=F.cross_entropy(all_preds_new,all_y)
        loss_new=classifier_loss_new+disc_loss_new
        loss=loss_new+loss_origin
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': loss.item(), 'class_o': classifier_loss_origin.item(),'class_n': classifier_loss_new.item(), 'dis_o': disc_loss_origin.item(),'dis_n': disc_loss_new.item()}

    def predict(self, x):
        x=self.featurizer.first_layer(x)
        x = self.avgpool(x)
        # b,64,16,16
        x = x.view(x.size(0), x.size(1), -1)
        # b,64,256
        common_embedding_batch = self.common_embedding.repeat(x.size(0), 1, 1)

        x = self.cross_attn_layer(x, common_embedding_batch)
        # b,64,256
        x = x.view(x.size(0), x.size(1), 16, 16)
        # b,64,16,16
        x = self.featurizer.last_four_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck_n(x)
        return self.classifier_n(x)

    def feature(self,x):
        x = self.featurizer.first_layer(x)
        x = self.avgpool(x)
        # b,64,16,16
        x = x.view(x.size(0), x.size(1), -1)
        # b,64,256
        common_embedding_batch = self.common_embedding.repeat(x.size(0), 1, 1)

        x = self.cross_attn_layer(x, common_embedding_batch)
        # b,64,256
        x = x.view(x.size(0), x.size(1), 16, 16)
        # b,64,16,16
        x = self.featurizer.last_four_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck_n(x)
        return x





class CrossAttentionEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CrossAttentionEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CrossAttentionEncoderLayer, self).__setstate__(state)

    def forward(self, x: Tensor,y:Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            x: the q sequence to the encoder layer (required).
            y: the k v sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if self.norm_first:
            raise NotImplementedError
        else:
            x = self.norm1(x + self._ca_block(x,y, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # cross attention block
    def _ca_block(self, x,y,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, y, y,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))