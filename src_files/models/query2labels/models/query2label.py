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

# Modified HIDDEN_DIM to include DINOv3 architectures
HIDDEN_DIM = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'tresnetl': 2432,
    'tresnetxl': 2656,
    'tresnetl_v2': 2048,
    'swin_l_224_22k':1536,
    'dinov3_vitb16_pretrain': 768,
    'dinov3_vitl16_pretrain': 1024,
}

def add_q2l_args(args):

    args.pretrained = True
    args.enc_layers = 1
    args.dec_layers = 2
    args.dim_feedforward = 8192
    # The backbone name is now correctly resolved before this function is called.
    if args.backbone not in HIDDEN_DIM:
        raise ValueError(f"Backbone '{args.backbone}' not found in HIDDEN_DIM mapping.")
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



# # query2label.py
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
    def __init__(self, backbone, transfomer, num_class, precomputed_query_embed=None):
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        # 定义输入投影层
        hidden_dim = transfomer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
        # --- 核心集成点 ---
        if precomputed_query_embed is not None:
            # 使用 nn.Embedding.from_pretrained 来初始化
            # freeze=False 意味着这个嵌入层将在训练中被微调
            self.query_embed = nn.Embedding.from_pretrained(precomputed_query_embed, freeze=False)
            print("Query embeddings initialized from CLIP.")
        else:
            # 如果没有提供预训练嵌入，则随机初始化
            self.query_embed = nn.Embedding(num_class, hidden_dim)
            print("Query embeddings initialized randomly.")

        # # 步骤 1: 创建原始的标签嵌入 (query_embed) 和一个用于掩码的特殊嵌入 (masked_query_embed)
        # self.query_embed = nn.Embedding(num_class, hidden_dim)
        
        # 这个 [MASK] 嵌入是可学习的参数，模型会学着用它来表示“未知标签”
        self.masked_query_embed = nn.Parameter(torch.Tensor(hidden_dim))
        nn.init.xavier_uniform_(self.masked_query_embed.unsqueeze(0)) # 良好地初始化

        # 定义最后的分类层
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        # 注意：静态的、基于规则的 label_mask 在此方案中不再需要，可以注释或删除
        # self.register_buffer("label_mask", self.build_label_mask(label_names))

    # forward 方法现在需要接收真实标签 `targets` 以便进行掩码
    def forward(self, input, targets=None):
        """
        Args:
            input (torch.Tensor): 输入的图像张量, 形状 [bs * num_patches, C, H, W]
            targets (torch.Tensor, optional): 形状为 [bs, num_class] 的真实标签. 
                                            只在训练时需要. Defaults to None.
        """
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        src = self.input_proj(src)
        # total_bs 是包含了 patches 的总 batch size, e.g., 40
        total_bs = src.shape[0] 

        # 准备送入 Transformer 的 Query 输入
        # 形状: [num_class, hidden_dim] -> [num_class, 1, hidden_dim] -> [num_class, total_bs, hidden_dim]
        query_input = self.query_embed.weight.unsqueeze(1).repeat(1, total_bs, 1)

        # MASK-II 策略核心逻辑
        if self.training:
            assert targets is not None, "Targets must be provided during training for MASK-II."
            
            # original_bs 是原始的 batch size, e.g., 8
            original_bs = targets.shape[0]
            
            # 计算每个原始样本被分成了多少份 (patches)
            num_patches = total_bs // original_bs
            assert total_bs % original_bs == 0, "Total batch size must be a multiple of original batch size."

            # --- 核心修改 ---
            # 将 targets 从 [original_bs, num_class] 扩展到 [total_bs, num_class]
            # [8, 5] -> [8, 1, 5] -> [8, 5, 5] -> [40, 5]
            # 使用 repeat_interleave 更直观
            expanded_targets = torch.repeat_interleave(targets, num_patches, dim=0)

            mask_prob = 0.25
            # 使用扩展后的 targets 来创建掩码，其形状将是 [total_bs, num_class]
            rand_mask = torch.rand(expanded_targets.shape, device=expanded_targets.device) < mask_prob
            actual_mask = rand_mask & expanded_targets.bool()

            # 广播和替换逻辑保持不变，因为现在的 actual_mask 维度已经正确
            # actual_mask 形状: [40, 5]
            # broadcast_mask 形状: [5, 40, 1]
            broadcast_mask = actual_mask.T.unsqueeze(-1)
            
            masked_embed_b = self.masked_query_embed.unsqueeze(0).unsqueeze(0)
            
            # query_input 形状: [5, 40, 2048]
            # 现在所有张量的维度都匹配了
            query_input = torch.where(broadcast_mask, masked_embed_b, query_input)

        # 将处理后的 query 送入 Transformer
        hs = self.transformer(
            src, 
            query_input, 
            pos,
            mask=None
        )[0]
        
        out = self.fc(hs[-1])
        
        # 注意：输出 out 的形状现在是 [total_bs, num_class]，即 [40, 5]
        # 在计算损失函数时，你需要使用 expanded_targets
        # 但通常情况下，更常见的做法是在这里将输出聚合起来
        # 例如，取平均值，使其变回 [original_bs, num_class]
        # out = out.view(original_bs, num_patches, self.num_class).mean(dim=1)
        
        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))

def build_q2l(args):
    # Safely parse backbone name from model_name
    if args.model_name.startswith('q2l_'):
        args.backbone = args.model_name[4:]
    else:
        args.backbone = args.model_name
    
    add_q2l_args(args)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    # 定义天气标签名称（需要与数据顺序一致）
    weather_labels = ["Sunny", "Cloudy", "Foggy", "Rainy", "Snowy"]  

     # 1. 实例化 CLIPLabelEncoder
    clip_encoder = CLIPLabelEncoder(
        label_names=weather_labels,
        hidden_dim=args.hidden_dim,
        clip_model_name=args.clip_model_name
    )

    # 2. 获取预计算的嵌入向量
    with torch.no_grad():
        precomputed_embeds = clip_encoder()

    # 3. 将预计算的嵌入传入模型
    model = Qeruy2Label(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_classes,
        precomputed_query_embed=precomputed_embeds  # <--- 在这里传入
    )


    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    
    return model

