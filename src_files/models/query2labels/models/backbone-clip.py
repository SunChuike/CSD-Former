# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
We borrow the positional encoding from Detr and adding some other backbones.
"""
from collections import OrderedDict
import os
import warnings

import torch
from torch.functional import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import clip
import torch.nn.functional as F

from .. import models
from .cls_cvt import build_CvT
from .swin_transformer import build_swin_transformer

from ..utils.misc import clean_state_dict

from .position_encoding import build_position_encoding

from urllib.request import urlopen
from PIL import Image
import timm

# Import CLIP
# try:
#     import clip
# except ImportError:
#     print("clip library is not found, please install with `pip install git+https://github.com/openai/CLIP.git`")



def get_model_path(modelname, pretrain_type, pretrain_dir):
    """
        Config your pretrained model path here!
    """
    # raise NotImplementedError("Please config your pretrained modelpath!")
    PTDICT = {
        'tresnetl_in1k': 'tresnet_l_448.pth',
        'tresnetl_v2_in21k': 'tresnet_l_v2_miil_21k.pth',
        'tresnetl_v2_oi': 'tresnet_l_pretrain_ml_decoder.pth',
        
    }
    return os.path.join(pretrain_dir, PTDICT[modelname+'_'+pretrain_type]) 



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layers: Dict=None
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        if return_layers is not None:
             self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
             self.body = backbone
        self.num_channels = num_channels
    
    def forward(self, input: Tensor):
        if isinstance(self.body, IntermediateLayerGetter):
            xs = self.body(input)
            out: Dict[str, Tensor] = {}
            for name, x in xs.items():
                 out[name] = x
            return out
        else:
           # print('11111')
           return self.body(input)



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrain_type: str,
                 pretrain_dir: str
                 ):
        self.backbone_name = name.lower()
        
        NCDICT = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'tresnetl': 2432,
            'tresnetxl': 2656,
            'tresnetl_v2': 2048,
            'clip':512,
        }
        num_channels = NCDICT[name]
        backbone = None
        return_layers = None # Initialize to avoid UnboundLocalError

        if name in ['resnet18', 'resnet50', 'resnet34', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True,
                norm_layer=FrozenBatchNorm2d)
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        elif name in ['tresnetl', 'tresnetxl', 'tresnetl_v2']:
            backbone = getattr(models, name)(
                {'num_classes': 1}
            )
            # load pretrained model
            if pretrain_type != '' and pretrain_type != None:
                pretrainedpath = get_model_path(name, pretrain_type, pretrain_dir)
                checkpoint = torch.load(pretrainedpath, map_location='cpu')
                from collections import OrderedDict
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head.fc' not in k})
                backbone.load_state_dict(_tmp_st, strict=False)
            
            if return_interm_layers:
                raise NotImplementedError('return_interm_layers must be False in TResNet!')
            return_layers = {'body': "0"}
        elif name in ['clip']:
            clip_model = clip.load('ViT-B/32', device="cuda", jit=False)[0]
            backbone = clip_model.visual
        else:
            raise NotImplementedError("Unknow name: %s" % name)
            
        
        super().__init__(backbone, train_backbone, num_channels, return_layers)   

    def forward(self, input: Tensor):
        if self.backbone_name in ['clip']:
            return super().forward(input)


         


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args=None):
        super().__init__(backbone, position_embedding)
        # self.args = args
        if args is not None and 'interpotaion' in vars(args) and args.interpotaion:
            self.interpotaion = True
        else:
            self.interpotaion = False


    def forward(self, input: Tensor):
        xs = self[0](input)
        out: List[Tensor] = []
        pos = []
        if isinstance(xs, dict):
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.dtype))
        elif isinstance(xs, torch.Tensor):  # Handle tensor output
            print(f"xs is a tensor with shape: {xs.shape}")  # Debugging: print shape of the tensor
            # print("mask_ratio:", self.mask_ratio, "type:", type(self.mask_ratio))
            # 1. 使用 unsqueeze 添加空间维度
            xs = xs.unsqueeze(2).unsqueeze(3)  # 变成 [40, 512, 1, 1]
            # 2. 使用 repeat 将空间维度扩展为 7x7
            xs = xs.repeat(1, 1, 7, 7)
            # # 3. 随机掩码
            # mask = random_mask(input.shape, self.mask_ratio)
            # # 应用掩码
            # masked_input = input * mask
            # # 经过backbone后
            # masked_xs = self[0](masked_input)
            # masked_xs = masked_xs.unsqueeze(2).unsqueeze(3)  # 变成 [40, 512, 1, 1]
            # # 2. 使用 repeat 将空间维度扩展为 7x7
            # masked_xs = masked_xs.repeat(1, 1, 7, 7)
            # # print("masked_xs:", masked_xs.shape)
            # # 将特征拼接在一起
            # fused_feature = torch.cat([xs, masked_xs], dim=1)

            # # 使用 1x1 卷积层来学习融合权重
            # fusion_weight = self.fusion_weight(fused_feature)

            # # 使用权重融合，也可以使用其他方式例如注意力
            # xs = (1 - torch.sigmoid(fusion_weight)) * xs + torch.sigmoid(fusion_weight) * masked_xs
            # # print('xs.shape', xs.shape)

            # # # 1. 使用 unsqueeze 添加空间维度
            # # xs = xs.unsqueeze(2).unsqueeze(3)  # 变成 [40, 512, 1, 1]
            # # # 2. 使用 repeat 将空间维度扩展为 7x7
            # # xs = xs.repeat(1, 1, 7, 7)
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        else:
            # for swin Transformer
            out.append(xs)
            pos.append(self[1](xs).to(xs.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = True
    if args.backbone in ['swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        imgsize = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone, imgsize)
        if args.pretrained:
            pretrainedpath = get_model_path(args.backbone)
            checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
            from collections import OrderedDict
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head' not in k})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        backbone.forward = backbone.forward_features
        bb_num_channels = backbone.embed_dim * 8
        del backbone.avgpool
        del backbone.head
    elif args.backbone in ['CvT_w24']:
        backbone = build_CvT(args.backbone, args.num_class)
        if args.pretrained:
            pretrainedpath = get_model_path(args.backbone)
            checkpoint = torch.load(pretrainedpath, map_location='cpu')
            from collections import OrderedDict
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if 'head' not in k})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        bb_num_channels = backbone.dim_embed[-1]
        backbone.forward = backbone.forward_features
        backbone.cls_token = False
        del backbone.head
    else:
        return_interm_layers = False
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, False, args.pretrain_type, args.pretrain_dir)
        bb_num_channels = backbone.num_channels
    model = Joiner(backbone, position_embedding, args)
    model.num_channels = bb_num_channels
    return model