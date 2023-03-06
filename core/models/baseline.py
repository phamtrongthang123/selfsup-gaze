import torch.nn as nn
import torch.nn.functional as F

from ..utils import getter
from .baseline_utils import *
__all__ = ['SelfSupervisedGaze']


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
        self.learnable_pe = nn.Embedding(config['image_size'],config["hidden_size"])

    def forward(self, x):
        
