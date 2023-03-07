import torch.nn as nn
import torch.nn.functional as F

from ..utils import getter
from .baseline_utils import *
__all__ = ['SelfSupervisedGaze']
def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class SelfSupervisedGaze(nn.Module):
    """_summary_

    There is two training mode for this autoregression: regression and classification. 
    Z-> X
    |
    v
    Y
    If we consider x,y ~ N then it is regression, else classification.

    Args:
        nn (_type_): _description_
    """    
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.device = config['device']
        self.fixation_encoder = FixationEncoderPE2D(config)
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
        # add 3 to avoid unnesscary cuda assert error 
        # *2 because we need xy
        self.decoder_word = nn.Linear(
            config["hidden_size"], (1000)*2
        )  # if use cross entropy loss from pytorch, no need for log softmax


    def forward(self, fixation, fix_masks,):
        tgt_mask = generate_square_subsequent_mask(fixation.shape[1]).to(self.device)
        # embedding for fixation
        fix_feature = self.fixation_encoder(
            fixation, fix_masks
        )  # torch.Size([1, 400, 512])
        intermidiate_output = self.transformer_decoder(
            fix_feature, fix_feature, tgt_mask=tgt_mask
        )
        output = self.decoder_word(intermidiate_output)
        output = einops.rearrange(output, 'b l (d h) -> b l d h', d = 2)
        return output 
    
    def build_loss(self, pred, target, mask):
        """This loss is for classification scheme

        Args:
            pred (_type_): _description_
            target (_type_): _description_
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """        
        target = einops.rearrange(target, "b s l -> b s l")  # torch.Size([1, 400, 2])
        mask = einops.rearrange(mask, "b s l -> b s l")  # torch.Size([1, 400, 1])
        gt_number_sent = target.shape[1]
        pred_px = pred[:,:gt_number_sent, 0]
        pred_py = pred[:,:gt_number_sent, 1]
        gt_px = target[:,:,:1]
        gt_py = target[:,:,1:]
        N, T, V = pred_px.shape

        x_flat_px = pred_px.reshape(N * T , V)
        x_flat_py = pred_py.reshape(N * T , V)
        # make sure it is a class.
        y_flat_px = gt_px.reshape(N * T ).long()
        y_flat_py = gt_py.reshape(N*T).long()
        mask_flat = mask.reshape(N * T)

        loss_px = torch.nn.functional.cross_entropy(
            x_flat_px, y_flat_px, reduction="none", label_smoothing=0.1
        )

        loss_py = torch.nn.functional.cross_entropy(
            x_flat_py, y_flat_py, reduction="none", label_smoothing=0.1
        )
        loss_px = torch.mul(loss_px, mask_flat)
        loss_py = torch.mul(loss_py, mask_flat)

        loss = torch.mean(loss_px) + torch.mean(loss_py)
        return loss