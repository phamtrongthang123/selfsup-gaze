from functorch import vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import pathlib
import pickle
import einops
import math

import torchvision


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
        """
        # x = x * math.sqrt(self.d_model)
        if len(x.shape) == 2:
            x = x + self.pe[: x.shape[0]]
        else:
            x = x + self.pe[: x.shape[1]].view(1, -1, self.d_model)
        return self.dropout(x)

from torchvision.models import resnet50, ResNet50_Weights
class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_encoder = torchvision.models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )  # shape out = 512 for resnet 18
        self.img_mapping_to_hidden = nn.Sequential(
            nn.Linear(self.img_encoder.fc.in_features, config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
        )  # make it always to hidden size
        self.img_encoder.fc = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
        """
        img_features = self.img_mapping_to_hidden(
            self.img_encoder(x)
        )  # torch.Size([1, 512])
        img_features = einops.rearrange(
            img_features, "b l -> b () l"
        )  # torch.Size([1, 1, 512])
        return img_features






class FixationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fixation_embed = nn.Sequential(
            nn.Linear(3, config["hidden_size"]),
            nn.GELU(),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
        )
        self.att_fixation = nn.MultiheadAttention(
            config["hidden_size"],
            config["num_attention_heads"],
            dropout=config["attention_probs_dropout_prob"],
            batch_first=True,
        )
        self.fix_mlp = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
        )
        self.fixnorm_mlp = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
        )
        self.pe = PositionalEncoding(
            config["hidden_size"], dropout=config["hidden_dropout_prob"]
        )
        self.norm_fixation = nn.LayerNorm(config["hidden_size"])

    def forward(self, fixation, fix_masks):
        """
        Args:
            x: Tensor, shape [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
        """

        fix_masks = einops.rearrange(
            fix_masks, "b s l -> b (s l)"
        )  # torch.Size([1, 400]), last dim is always 1 anyway.
        fix_masks_tril = self.__make_att_mask_for_fixation(
            fix_masks
        )  # torch.Size([bs, 400, 400])
        
        fix_feature = self.fixation_embedding(fixation)  # torch.Size([bs, 400, 512])
        # fix_feature = einops.rearrange(
        #     fix_feature, "(b s) l -> b s l", b=1
        # )  # torch.Size([1, 400, 512])
        fix_feature = self.fix_mlp(fix_feature)
        fix_feature_att = self.att_fixation(
            fix_feature, fix_feature, fix_feature, attn_mask=fix_masks_tril
        )[
            0
        ]  # torch.Size([1, 400, 512])
        fix_feature_norm = self.norm_fixation(fix_feature_att + fix_feature)
        fix_feature_norm = self.fixnorm_mlp(
            fix_feature_norm
        )  # torch.Size([1, 400, 512])
        return fix_feature_norm
    
    def fixation_embedding(self, x):
        x =  self.fixation_embed(x)
        # fix_feature = einops.rearrange(
        #     x, "b s l -> (b s) l"
        # )  # torch.Size([400, 512])
        fix_feature = self.pe(fix_feature)
        return fix_feature

    def __make_att_mask_for_fixation(self, fix_masks):
        tgt_mask = torch.tril(
            torch.ones(
                fix_masks.shape[0],
                fix_masks.shape[1],
                fix_masks.shape[1],
                device=fix_masks.device,
            )
        )  # [num_sent, length, length]
        masking = vmap(
            lambda mask1, mask2: mask1.masked_fill_(mask2 == 0.0, 0), in_dims=(0, 0)
        )
        # tgt_mask = torch.tril(
        #     torch.ones(fix_masks.shape[1], fix_masks.shape[1], device=fix_masks.device)
        # )  # [length, length]
        # final_mask = tgt_mask.masked_fill_(fix_masks == 0.0, 0)  # [length, length]
        final_mask = masking(tgt_mask, fix_masks)  # [num_sent, length, length]
        final_mask = (1 - final_mask) * -1e9
        final_mask = einops.repeat(
            final_mask, "b s l -> (b n) s l", n=self.config["num_attention_heads"]
        )  # torch.Size([head*bs, 400, 400]), assume the batchsize include the number of sentences
        # [length, length]
        return final_mask

class FixationEncoderPE2D(FixationEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.fixation_embed_x = nn.Embedding(1000, config["hidden_size"]) # could go to 225 or more, set to 1000 to make it safer
        self.fixation_embed_y = nn.Embedding(1000, config["hidden_size"])

    def fixation_embedding(self, x):
        x_axis = x[:, :, 0].long()
        y_axis = x[:, :, 1].long()
        duration = x[:, :, 2]
        x = self.pe(self.fixation_embed_x(x_axis)) + self.pe(self.fixation_embed_y(y_axis)) * duration.unsqueeze(-1)
        # x = einops.rearrange(
        #     x, "b s l -> (b s) l"
        # )  # torch.Size([400, 512])
        return x
    

class ImageFixationFuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att_fused_img_fix = nn.MultiheadAttention(
            config["hidden_size"],
            config["num_attention_heads"],
            dropout=config["attention_probs_dropout_prob"],
            batch_first=True,
        )
        self.norm_fused_img_fix = nn.LayerNorm(config["hidden_size"])
        self.fused_mlp = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
        )
        self.double_pe = DoublePE(config)

    def forward(self, img_features, fix_feature, length):
        """
        Args:
            x: Tensor, shape [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
        """
        # fusing between img and fixation
        fused_img_fix = self.att_fused_img_fix(fix_feature, img_features, img_features)[
            0
        ]  # torch.Size([1, 400, 512])
        fused_img_fix_norm = self.norm_fused_img_fix(fused_img_fix + fix_feature)
        # repeat to match the shape of caption
        fused_img_fix_norm = einops.repeat(
            fused_img_fix_norm, "b s d -> (l b) s d", l=length
        )  # torch.Size([3, 400, 512])
        fused_img_fix_norm = self.fused_mlp(fused_img_fix_norm)
        fused_img_fix_norm = self.double_pe(fused_img_fix_norm)
        return fused_img_fix_norm


class DoublePE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pe = PositionalEncoding(
            config["hidden_size"], dropout=config["hidden_dropout_prob"]
        )

    def forward(self, x):
        x = self.pe(x)
        x = einops.rearrange(x, "n l d -> l n d")
        x = self.pe(x)
        x = einops.rearrange(x, "n l d -> l n d")
        return x


class CaptionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.double_pe = DoublePE(config)
        self.word_embed = nn.Embedding(self.vocab_size + 1, config["hidden_size"])

    def forward(self, captions, cap_masks):
        captions = einops.rearrange(captions, "b s l -> (b s) l")  # torch.Size([3, 50])
        cap_feature = self.word_embed(captions)  # torch.Size([3, 50, 512])
        cap_feature = self.double_pe(cap_feature)

        cap_masks_tril = self.__make_att_mask_for_caption(
            cap_masks
        )  # torch.Size([3, 50, 50])
        # decoding
        cap_masks_tril = einops.repeat(
            cap_masks_tril, "b s l -> (b n) s l", n=self.config["num_attention_heads"]
        )  # torch.Size([3, 50]), assume the batchsize include the number of sentences
        return cap_feature, cap_masks_tril

    def __make_att_mask_for_caption(self, cap_masks):
        # cap_masks torch.Size([1, 3, 50]), and we want the mask to be [3, 50, 50]

        cap_masks = einops.rearrange(
            cap_masks, "b x y -> (b x) y"
        )  # assume batch always = 1
        tgt_mask = torch.tril(
            torch.ones(
                cap_masks.shape[0],
                cap_masks.shape[1],
                cap_masks.shape[1],
                device=cap_masks.device,
            )
        )  # [num_sent, length, length]
        masking = vmap(
            lambda mask1, mask2: mask1.masked_fill_(mask2 == 0.0, 0), in_dims=(0, 0)
        )
        final_mask = masking(tgt_mask, cap_masks)  # [num_sent, length, length]
        final_mask = (1 - final_mask) * -1e9
        # [num_sent, length, length]
        return final_mask
