from itertools import chain
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math
import clip

from .backbone import build_backbone
from .transformer import build_transformer
from ..utils.misc import clean_state_dict

# --- 全局常量 ---
VERY_NEGATIVE_NUMBER = -1e9

HIDDEN_DIM = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'tresnetl': 2432,
    'tresnetxl': 2656,
    'tresnetl_v2': 2048,
    'swin_l_224_22k': 1536
}


def add_q2l_args(args):
    """
    为 Query2Label 模型添加和设置必要的 Transformer 参数。
    这个函数假设 `args.backbone` 已经被确定。
    """
    args.pretrained = True
    args.enc_layers = 1
    args.dec_layers = 2
    args.dim_feedforward = 8192
    args.dropout = 0.1
    args.nheads = 4
    args.pre_norm = False
    args.position_embedding = 'sine'
    args.keep_other_self_attn_dec = False
    args.keep_first_self_attn_dec = False
    args.keep_input_proj = False

    # 根据 args.backbone 设置 args.hidden_dim
    if args.backbone in HIDDEN_DIM:
        args.hidden_dim = HIDDEN_DIM[args.backbone]
    else:
        raise ValueError(f"Backbone '{args.backbone}' not found in HIDDEN_DIM mapping. Please update HIDDEN_DIM or check the backbone name.")


class GroupWiseLinear(nn.Module):
    """Group-wise linear layer for multi-label classification."""
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
        # x: B, K, d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class CLIPLabelEncoder(nn.Module):
    """Encodes text labels into semantic embeddings using CLIP."""
    def __init__(self, label_names, hidden_dim, clip_model_name="ViT-B/32"):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model_name)
        self.clip_model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.texts = [
            "A sunny day is characterized by a clear, azure blue sky with minimal cloud cover, allowing sunlight to brightly illuminate the scene and create vibrant colors.",
            "A cloudy day is defined by thick layers of clouds obscuring the sky, resulting in soft, diffused light and potentially occasional rays peeking through the cloud cover.",
            "A foggy day is marked by reduced visibility due to a dense mist in the air, causing distant objects to appear blurred and creating a damp atmosphere.",
            "A rainy day is identified by a drizzle or rainfall that dampens the ground, causes surfaces to reflect light, and fills the air with a moist, refreshing atmosphere.",
            "A snowy day transforms the landscape with a blanket of white snow covering the ground, trees, and houses, accompanied by snowflakes gently drifting through the air, creating a picturesque winter scene."
        ]

        self.text_inputs = torch.cat([clip.tokenize(text) for text in self.texts]).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        self.proj = nn.Linear(text_features.shape[-1], hidden_dim).to(self.device)
        self.init_weights(text_features)

    def init_weights(self, text_features):
        self.proj.weight.data.normal_(mean=0.0, std=0.02)
        with torch.no_grad():
            projected = self.proj(text_features.float())
            projected /= projected.norm(dim=-1, keepdim=True)
            self.proj.bias.data = -self.proj.weight.data.mean(dim=1)

    def forward(self):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return self.proj(text_features.float())


class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class, label_names, clip_model_name, learnable_mask, hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        self.learnable_mask_enabled = learnable_mask  # Store the boolean flag

        self.label_encoder = CLIPLabelEncoder(label_names, hidden_dim, clip_model_name)
        self.device = self.label_encoder.device

        # Initialize and move mask to the correct device
        if self.learnable_mask_enabled:
            initial_mask = self.build_label_mask(label_names, use_hard_masking=True)
            self.learnable_label_mask = nn.Parameter(initial_mask.float().to(self.device))
        else:
            initial_mask = torch.zeros(num_class, num_class)
            self.register_buffer("static_label_mask", initial_mask.float().to(self.device))
            self.learnable_label_mask = None # Explicitly set to None

        # Initialize other components and move them to the device
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1).to(self.device)
        
        self.query_embed = nn.Embedding.from_pretrained(
            self.label_encoder(), freeze=False
        ).to(self.device)
        self.query_embed.weight.requires_grad = True

        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True).to(self.device)

    @staticmethod
    def build_label_mask(label_names, use_hard_masking=True):
        label_relations = {
            "Sunny": ["Cloudy"],
            "Cloudy": ["Foggy", "Rainy", "Snowy"],
            "Foggy": ["Cloudy", "Rainy"],
            "Rainy": ["Cloudy", "Snowy"],
            "Snowy": ["Cloudy", "Rainy"]
        }
        n = len(label_names)

        if use_hard_masking:
            mask = torch.full((n, n), VERY_NEGATIVE_NUMBER)
            for i, name1 in enumerate(label_names):
                mask[i, i] = 0.0
                for j, name2 in enumerate(label_names):
                    if name2 in label_relations.get(name1, []):
                        mask[i, j] = 0.0
        else:
            mask = torch.zeros(n, n)
        return mask

    def forward(self, input):
        # Ensure input data is on the correct device
        input = input.to(self.device)

        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        src = self.input_proj(src)
        query_input = self.query_embed.weight

        mask_to_pass = self.learnable_label_mask if self.learnable_mask_enabled else self.static_label_mask
        
        hs = self.transformer(
            src,
            query_input,
            pos,
            mask=None,
            label_mask=mask_to_pass
        )[0]
        
        decoder_output = hs[-1]
        out = self.fc(decoder_output)
        return out

    def finetune_paras(self):
        params = [
            self.transformer.parameters(),
            self.fc.parameters(),
            self.input_proj.parameters(),
            self.query_embed.parameters()
        ]
        if self.learnable_mask_enabled and self.learnable_label_mask is not None:
            # nn.Parameter is not a list, so we add it directly
            params.append(self.learnable_label_mask)
        return chain(*params)

    def load_backbone(self, path):
        print(f"=> loading checkpoint '{path}'")
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print(f"=> loaded checkpoint '{path}' (epoch {checkpoint.get('epoch', 'N/A')})")


def build_q2l(args):
    """Builds the complete Query2Label model."""
    # 1. Check for required arguments
    if not hasattr(args, 'model_name'):
        raise ValueError("args.model_name is required for build_q2l")
    if not hasattr(args, 'num_classes'):
        raise ValueError("args.num_classes is required for build_q2l")
    if not hasattr(args, 'clip_model_name'):
        args.clip_model_name = "ViT-B/32"  # Provide a default

    # 2. Determine backbone from model_name or directly
    if args.model_name.startswith('q2l_'):
        args.backbone = args.model_name[len('q2l_'):]
    elif not hasattr(args, 'backbone') or not args.backbone:
        raise ValueError("If model_name doesn't start with 'q2l_', args.backbone must be provided.")
    
    # 3. Set up Q2L-specific arguments, including hidden_dim
    add_q2l_args(args)
    
    # 4. Build components
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    weather_labels = ["Sunny", "Cloudy", "Foggy", "Rainy", "Snowy"]
    enable_learnable_mask = getattr(args, 'learnable_mask', False)

    # 5. Instantiate the model
    model = Qeruy2Label(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_classes,
        label_names=weather_labels,
        clip_model_name=args.clip_model_name,
        learnable_mask=enable_learnable_mask,
        hidden_dim=args.hidden_dim
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("Set model.input_proj to Identity!")

    return model