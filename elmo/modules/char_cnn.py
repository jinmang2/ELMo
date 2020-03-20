# -*- coding: utf-8 -*-

# Python Standard Library
import logging

# Python Installed Library
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# User Defined Library
from .highway import Highway


class ConvTokenEmbedder(nn.Module):
    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda):
        super(ConvTokenEmbedder, self).__init__()
        self.config = config
        self.use_cuda = use_cuda

        self.word_emb_layer = word_emb_layer
        self.char_emb_layer = char_emb_layer

        self.output_dim = config['encoder']['projection_dim']
        self.emb_dim = 0
        if word_emb_layer is not None:
            self.emb_dim += word_emb_layer.n_d

        if char_emb_layer is not None:
            self.convolutions = []
            cnn_config = config['token_embedder']
            filters = cnn_config['filters']
            char_embed_dim = cnn_config['char_dim']

            for i, (width, num) in enumerate(filters):
                conv = torch.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num, # 문자를 몇 개나 볼 것인지
                    kernel_size=width,
                    bias=True
                )
                if use_cuda:
                    conv = conv.cuda()
                self.convolutions.append(conv)

            self.convolutions = nn.ModuleList(self.convolutions)

            self.n_filters = sum(f[1] for f in filters)
            self.n_highway = cnn_config['n_highway']

            self.highways = Highway(self.n_filters,
                                    self.n_highway,
                                    activation=F.relu,
                                    use_cuda=use_cuda)
            self.emb_dim += self.n_filters

        self.projection = nn.Linear(self.emb_dim, self.output_dim, bias=True)
        if use_cuda:
            self.projection = self.projection.cuda()

    def forward(self, word_inp, chars_inp, shape):
        embs = []
        batch_size, seq_len = shape
        if self.word_emb_layer is not None:
            batch_size, seq_len = word_inp.size(0), word_inp.size(1)
            word_emb = self.word_emb_layer(
                Variable(word_inp).cuda() if self.use_cuda else Variable(word_inp))
            embs.append(word_emb)

        if self.char_emb_layer is not None:
            chars_inp = chars_inp.view(batch_size * seq_len, -1)

            # shape (batch_size, max_char, char_dim)
            character_embedding = self.char_emb_layer(
                Variable(chars_inp).cuda() if self.use_cuda else Variable(chars_inp))

            # shape (batch_size, char_dim, max_char)
            character_embedding = torch.transpose(
                character_embedding, 1, 2)

            cnn_config = self.config['token_embedder']
            if cnn_config['activation'] in ['tanh', 'relu']:
                activation = getattr(F, cnn_config['activation'])
            else:
                raise Exception("Unknown activation")

            convs = []
            for i in range(len(self.convolutions)):
                convolved = self.convolutions[i](character_embedding)
                # (batch_size * sequence_length, n_filters for this width)
                convolved, _ = torch.max(convolved, dim=-1)
                convolved = activation(convolved)
                convs.append(convolved)
            char_emb = torch.cat(convs, dim=-1)
            char_emb = self.highways(char_emb)

            embs.append(char_emb.view(batch_size, -1, self.n_filters))

        token_embedding = torch.cat(embs, dim=2)

        return self.projection(token_embedding)
