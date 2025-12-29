import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
import clip

from .backbone import build_backbone
from .transformer import build_transformer
from ..utils.misc import clean_state_dict

HIDDEN_DIM = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'tresnetl': 2432,
    'tresnetxl': 2656,
    'tresnetl_v2': 2048,
    'swin_l_224_22k':1536
}

def add_q2l_args(args):

    args.pretrained = True
    args.enc_layers = 1
    args.dec_layers = 2
    args.dim_feedforward = 8192
    args.hidden_dim = HIDDEN_DIM[args.backbone]
    args.dropout = 0.1
    args.nheads = 4
    args.pre_norm = False
    args.position_embedding = 'sine'
    args.keep_other_self_attn_dec = False
    args.keep_first_self_attn_dec = False
    args.keep_input_proj = False

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x



# query2label.py
class CLIPLabelEncoder(nn.Module):
    def __init__(self, label_names, hidden_dim, clip_model_name="ViT-B/32"):
        super().__init__()
        # 加载CLIP模型
        self.clip_model, _ = clip.load(clip_model_name)
        self.clip_model.eval()  # 设置为评估模式，防止梯度更新

        # 确定使用的设备，并移动CLIP模型到该设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  #可以确定放到哪个gpu上
        self.clip_model.to(self.device)

        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 定义天气标签的详细描述（加入先验知识）
        self.texts = [
            "A sunny day is characterized by a clear, azure blue sky with minimal cloud cover, allowing sunlight to brightly illuminate the scene and create vibrant colors.",
            "A cloudy day is defined by thick layers of clouds obscuring the sky, resulting in soft, diffused light and potentially occasional rays peeking through the cloud cover.",
            "A foggy day is marked by reduced visibility due to a dense mist in the air, causing distant objects to appear blurred and creating a damp atmosphere.",
            "A rainy day is identified by a drizzle or rainfall that dampens the ground, causes surfaces to reflect light, and fills the air with a moist, refreshing atmosphere.",
            "A snowy day transforms the landscape with a blanket of white snow covering the ground, trees, and houses, accompanied by snowflakes gently drifting through the air, creating a picturesque winter scene."
        ]
        
        # 生成标签文本描述
        self.text_inputs = torch.cat([
            clip.tokenize(text) for text in self.texts
        ]).to(self.device)

        # 预计算文本特征
        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        # 投影到模型维度
        self.proj = nn.Linear(text_features.shape[-1], hidden_dim).to(self.device)
        # self.proj.requires_grad_(False)
        self.init_weights(text_features)
        
    def init_weights(self, text_features):
        # 用CLIP特征初始化投影权重
        self.proj.weight.data.normal_(mean=0.0, std=0.02)
        with torch.no_grad():
            projected = self.proj(text_features.float())
            # 保持初始输出的范数与原始CLIP特征相似
            projected /= projected.norm(dim=-1, keepdim=True)
            self.proj.bias.data = -self.proj.weight.data.mean(dim=1)
            
    def forward(self):
        text_features = self.clip_model.encode_text(self.text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)       
        return self.proj(text_features.float())
    
class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class, label_names, clip_model_name,learnable_mask=False):
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        if learnable_mask:
                    self.label_mask = nn.Parameter(torch.ones(num_class, num_class))
        else:
                    mask = torch.tensor([
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1],
                        [0, 1, 0, 1, 1]
                    ], dtype=torch.bool)
                    self.register_buffer("label_mask", mask)

        # 定义输入投影层
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # 使用CLIP初始化标签嵌入
        self.label_encoder = CLIPLabelEncoder(label_names, hidden_dim, clip_model_name)

        self.query_embed = nn.Embedding.from_pretrained(
            self.label_encoder(), freeze=False)  # 允许微调
        self.query_embed.weight.requires_grad = True #确定embedding可以训练

        self.device = self.label_encoder.device
        self.query_embed.to(self.device)

        # 定义标签关系掩码矩阵 (0表示屏蔽，1表示保留)
        self.register_buffer("label_mask", self.build_label_mask(label_names))
        self.label_mask = self.label_mask.to(self.device)  # 移动到正确的设备

        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)
        self.fc.to(self.device) # 放到cuda上

    @staticmethod
    def build_label_mask(label_names):
        """根据天气标签的语义关系构建一个硬掩码矩阵"""
        label_relations = {
            "Sunny": ["Cloudy"],
            "Cloudy": ["Foggy", "Rainy", "Snowy"],
            "Foggy": ["Cloudy", "Rainy"],
            "Rainy": ["Cloudy", "Snowy"],
            "Snowy": ["Cloudy", "Rainy"]
        }
        # 初始化为负无穷，默认全部屏蔽
        mask = torch.full((len(label_names), len(label_names)), float('-inf'))
        
        for i, name in enumerate(label_names):
            # 允许关注自身
            mask[i, i] = 0.0
            # 允许关注合理关联的标签
            for j, other in enumerate(label_names):
                if other in label_relations.get(name, []):
                    mask[i, j] = 0.0
        
        # PyTorch的MultiheadAttention期望的mask格式是：
        # 0.0 表示不屏蔽
        # -inf 表示屏蔽
        return mask

    def forward(self, input, label_mask=None):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        src = self.input_proj(src)

        query_input = self.query_embed.weight
        # 在decoder中传入标签关系掩码
        hs = self.transformer(
            src, 
            query_input, 
            pos,
            mask = None,
            label_mask=self.label_mask  # 传入掩码矩阵
        )[0]
        out = self.fc(hs[-1])
        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_q2l(args):
    args.backbone = args.model_name[4:]
    add_q2l_args(args)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    # 定义天气标签名称（需要与数据顺序一致）
    weather_labels = ["Sunny", "Cloudy", "Foggy", "Rainy", "Snowy"]  

    model = Qeruy2Label(
        backbone = backbone,
        transfomer = transformer,
        num_class = args.num_classes,
        label_names = weather_labels,  # 传入标签名称
        clip_model_name = args.clip_model_name
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    
    return model