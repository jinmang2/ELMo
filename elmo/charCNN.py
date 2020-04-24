# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNNLayer(nn.Module):

    def __init__(self,
                 word_emb=None,
                 char_emb=None,
                 n_highway=2,
                 output_dim=512,
                 activation=F.relu):
        super().__init__()

        assert activation in ['relu', 'tanh'], "Activation Error"

        self.word_emb = word_emb
        self.char_emb = char_emb

        self.emb_dim = 0
        if word_emb is not None:
            self.emb_dim += word_emb.n_d
        if char_emb is not None:
            self.convolutions = []
            for width, dim in self.char_emb.filters:
                conv = nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=dim,
                    kernel_size=width,
                    bias=True,
                )
                self.convolutions.append(conv)
            self.convolutions = nn.ModuleList(self.convolutions)
            self.highways = Highway(sum(dims), n_highway)
            # self.emb_dim = char_embed_dim
        self.activation = getattr(F, activation)
        self.projection = nn.Linear(self.emb_dim, output_dim, bias=True)

    def forward(self, x):
        convs = []
        for conv in self.convolutions):
            convolved = conv(x)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self.activation(convolved)
            convs.append(convolved)
        char_emb = torch.cat(convs, dim=-1)
        char_emb = self.highways(char_emb)
