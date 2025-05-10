# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

import clip

class CLIPVisualBackbone(nn.Module):
    def __init__(self, model_name="ViT-B/32", trainable=False, device="cuda:0"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.visual = self.clip_model.visual.float()  # Save visual module
        self.text = self.clip_model.transformer.float()  # Save text module

        if not trainable:
            for param in self.visual.parameters():
                param.requires_grad = False
            for param in self.text.parameters():
                param.requires_grad = False

        self.device = device
        self.num_channels = self.visual.output_dim  # e.g., 512 or 768

    def encode_text(self, text):
        """Encode text using CLIP's text encoder"""
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():  # We typically want to keep text encoding fixed
            text_features = self.text(text_tokens)[0]  # Get the [CLS] token features
        return text_features

    def forward(self, x, text_conditioning=None):
        """
        Args:
            x: Image tensor of shape [B, 3, H, W]
            text_conditioning: String for text conditioning
        """
        x = F.interpolate(x, size=(224, 224))
        target_dtype = next(self.visual.parameters()).dtype
        x = x.to(device=self.device, dtype=target_dtype)
        
        # Get image features
        image_features = self.visual(x)  # [B, 512]
        
        if text_conditioning is not None:
            # Get text features
            text_features = self.encode_text(text_conditioning)  # [1, 512]
            
            # Expand text features to match batch size
            text_features = text_features.expand(image_features.shape[0], -1)  # [B, 512]
            
            # Combine image and text features (you can choose different combination methods)
            # Method 1: Addition
            combined_features = image_features + text_features
            
            # Alternative methods you might consider:
            # Method 2: Concatenation and projection
            # combined_features = torch.cat([image_features, text_features], dim=1)  # [B, 1024]
            # You would need to add a projection layer to get back to 512 dims
            
            # Method 3: Element-wise multiplication
            # combined_features = image_features * text_features
            
            return {'0': combined_features}
        
        return {'0': image_features}


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
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

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, 
                 return_interm_layers: bool,
                 name: str):
        super().__init__()
        for name_, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
            if not train_backbone or 'layer2' not in name_ and 'layer3' not in name_ and 'layer4' not in name_:
                parameter.requires_grad_(False)
        self.name = name
        print("name", name)
        print("return_interm_layers", return_interm_layers)
        if name.startswith('resnet'):
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if name.startswith('resnet'):
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        elif name.startswith('clip'):
            self.body = backbone

        self.num_channels = num_channels

    def forward(self, tensor, text_conditioning=None):
        xs = self.body(tensor)
        return xs

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if name.startswith('resnet'):
            print(f'=================Using resnet {name}=================')
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        elif name.startswith('clip'):
            print(f'=================Using CLIP {name}=================')
            backbone = CLIPVisualBackbone(model_name=name.split('_')[1], trainable=train_backbone)
            num_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, name)

from .position_encoding import PositionEmbeddingSine

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, name="resnet18"):
        super().__init__(backbone, position_embedding)
        self.name = name
        self.position_embedding = position_embedding or PositionEmbeddingSine(num_pos_feats=128, normalize=True)

    def forward(self, tensor_list: NestedTensor, text_conditioning=None):
        xs = self[0](tensor_list, text_conditioning)
        out: List[NestedTensor] = []
        pos = []
        for i, (name, x) in enumerate(xs.items()):
            out.append(x)
            # print(f"\n\nx.shape: {x.shape}")  ##
            # position encoding  
            if "clip" not in self.name:
                pos_emb = self[1](x)  ## x.shape: torch.Size([32, 512, 12, 20]); pos_emb.shape: torch.Size([1, 512, 12, 20])
            else:
                B, C = x.shape  # Shape is [B, C], i.e., [32, 512]
                
                # Reshape x as [B, C, 1, 1] so position encoding can apply to it
                x_reshaped = x.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, C, 1, 1]
                
                # Apply position encoding (position embedding needs to be adjusted for flat input)
                pos_emb = self.position_embedding(x_reshaped)  # Now works as 2D

                # Now flatten back the position embedding if necessary (reshape it to [B, C])
                pos_emb = pos_emb.flatten(2).squeeze(-1).squeeze(-1)  # Shape: [B, C]

            # print(f"pos_emb.shape: {pos_emb.shape}\n\n") 
            pos.append(pos_emb.to(x.dtype))
        # print(f"out.shape: {len(out)}\n\n") 
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding, name=args.backbone)
    model.num_channels = backbone.num_channels
    return model
