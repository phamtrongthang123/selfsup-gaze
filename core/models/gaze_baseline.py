import einops
import torch.nn as nn
import torch.nn.functional as F
import torch
from ..utils import getter
import torchvision
from .baseline_utils import *
import json
from functorch import vmap

__all__ = ["GazeBaseline"]


class GazeBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        with open(config["vocab_path"], "r") as f:
            self.vocab = json.load(f)
        self.vocab_size = config["vocab_size"]
        self.max_number_sent = config["max_number_sent"]
        self.img_encoder = ImageEncoder(config)
        self.fixation_encoder = FixationEncoderPE2D(config)
        self.image_fixation_fusion = ImageFixationFuser(config)
        # multihead attention
        self.caption_embed = CaptionEmbedding(config)
        self.att_capin = nn.MultiheadAttention(
            config["hidden_size"],
            config["num_attention_heads"],
            dropout=config["attention_probs_dropout_prob"],
            batch_first=True,
        )

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            config["hidden_size"],
            config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"],
            dropout=config["hidden_dropout_prob"],
            batch_first=True,
        )
        self.norm_capout = nn.LayerNorm(config["hidden_size"])
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            config["num_hidden_layers"],
            norm=self.norm_capout,
        )
        # self.decoder_word = nn.Sequential(nn.Linear(config["hidden_size"], self.vocab_size), nn.LogSoftmax(dim=-1))
        self.decoder_word = nn.Linear(
            config["hidden_size"], self.vocab_size
        )  # if use cross entropy loss from pytorch, no need for log softmax

        self.pe = PositionalEncoding(
            config["hidden_size"], dropout=config["hidden_dropout_prob"]
        )
        self.double_pe = DoublePE(config)
        self.learnable_pe = nn.Embedding(self.max_number_sent,400* config["hidden_size"])

        self.number_prediction = nn.Sequential(nn.Linear(config["hidden_size"], config["hidden_size"]), nn.ReLU(), nn.Linear(config["hidden_size"], 1))

    def forward(self, img, fixation, fix_masks, captions, cap_masks):
        # img torch.Size([1, 3, 224, 224]) fixation torch.Size([1, 400, 3]) fix_masks torch.Size([1, 400, 1]) captions torch.Size([1, 3, 50]) cap_masks torch.Size([1, 3, 50])
        # because it is fixation, i keep the original shape: full size x 3, instead of split for each sentences

        # embedding to create "memory" for decoder
        # embedding for image
        img_features = self.img_encoder(img)  # torch.Size([1, 1, 512])

        # embedding for fixation
        fix_feature = self.fixation_encoder(
            fixation, fix_masks
        )  # torch.Size([1, 400, 512])

        # fusing between img and fixation
        fused_img_fix = self.image_fixation_fusion(
            img_features, fix_feature, length=captions.shape[1]
        )  # torch.Size([ 3, 400, 512])
        num = self.number_prediction(fused_img_fix[0].mean(0).unsqueeze(0))
        # learnable_pe = self.learnable_pe.weight.unsqueeze(0).view(self.max_number_sent, 400, -1)
        # fused_img_fix = fused_img_fix + learnable_pe[:captions.shape[1]]

        # embedding for caption
        cap_feature, cap_masks_tril = self.caption_embed(
            captions, cap_masks
        )  # torch.Size([3, 50, 512]) torch.Size([3*numhead, 50, 50])
        output = self.transformer_decoder(
            cap_feature, fused_img_fix, tgt_mask=cap_masks_tril
        )
        output_sent = self.decoder_word(output)  # torch.Size([3, 50, vocab_size])
        # torch.Size([3, 50, 512])
        tmp = torch.argmax(output_sent, dim=2)
        return output_sent, num

    def build_loss(self, pred, target, mask):
        # torch.Size([max_number_sent/gt_length, 50, vocab_size]) torch.Size([1, gt_length, 50]) torch.Size([1, gt_length, 50])
        # target = einops.rearrange(target, 'b s l -> (b s) l') # torch.Size([3, 50])
        # mask = einops.rearrange(mask, 'b s l -> (b s) l') # torch.Size([3, 50])
        # one_hot = torch.nn.functional.one_hot(target, self.config['vocab_size'])
        # gt_number_sent = target.shape[0]
        # output = -(one_hot * pred[:gt_number_sent] * mask[:, :, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        # return output.mean()
        target = einops.rearrange(target, "b s l -> (b s) l")  # torch.Size([3, 50])
        mask = einops.rearrange(mask, "b s l -> (b s) l")  # torch.Size([3, 50])
        gt_number_sent = target.shape[0]
        N, T, V = pred[:gt_number_sent].shape

        x_flat = pred[:gt_number_sent].reshape(N * T, V)
        y_flat = target.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(
            x_flat, y_flat, reduction="none", label_smoothing=0.1
        )
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)
        return loss

    def generate_greedy(self, img, fixation, fix_masks):
        # generate the caption with beam search
        # beam search implementation here
        # embedding to create "memory" for decoder
        # embedding for image

        img_features = self.img_encoder(img)  # torch.Size([1, 1, 512])

        # embedding for fixation
        fix_feature = self.fixation_encoder(
            fixation, fix_masks
        )  # torch.Size([1, 400, 512])

        # fusing between img and fixation
        fused_img_fix = self.image_fixation_fusion(
            img_features, fix_feature, self.max_number_sent
        )  # torch.Size([ maxnumsent, 50, 512])
        num = self.number_prediction(fused_img_fix[0].mean(0).unsqueeze(0))
        # learnable_pe = self.learnable_pe.weight.unsqueeze(0).view(self.max_number_sent, 400, -1)
        # fused_img_fix = fused_img_fix + learnable_pe
        # decoding
        # start with <sos> token
        # torch.Size([1, 1])
        cap_output = torch.tensor(
            [self.vocab["word2idx"]["<SOS>"]], device=fused_img_fix.device
        ).unsqueeze(0)
        cap_output = einops.repeat(
            cap_output, "b s ->() (l b) (s k)", l=self.max_number_sent, k=49
        )  # torch.Size([1, max_num_sent, 50])
        # mask
        cap_masks = torch.zeros(
            (1, cap_output.shape[1], 49),
            device=fused_img_fix.device,
            dtype=torch.float32,
        )  # torch.Size([1, max_num_sent, 50])
        probs_cap = None
        word_cap = None
        cap_masks[..., 0] = 1.0
        for i in range(49):
            cap_output = cap_output.clone()
            if i != 0:
                cap_output[..., i] = next_token
                cap_masks[..., i] = 1.0
            cap_feature, cap_masks_tril = self.caption_embed(cap_output, cap_masks)
            output = self.transformer_decoder(
                cap_feature, fused_img_fix, tgt_mask=cap_masks_tril
            )
            output_sent = self.decoder_word(
                output
            )  # torch.Size([max_num_sent, 50, vocab_size])

            next_token = torch.argmax(
                output_sent[:, i], dim=1
            )  # torch.Size([max_num_sent, 50])
            if probs_cap is None:
                probs_cap = output_sent[:, i].unsqueeze(1)
            else:
                probs_cap = torch.cat(
                    (probs_cap, output_sent[:, i].unsqueeze(1)), dim=1
                )
            if word_cap is None:
                word_cap = next_token.unsqueeze(1)
            else:
                word_cap = torch.cat((word_cap, next_token.unsqueeze(1)), dim=1)
        return word_cap, probs_cap, num
