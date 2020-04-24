# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):

    def __init__(self,
                 n_d,
                 word2id,
                 embs=None,
                 fix_emb=True,
                 normalize=True,
                 oov='[OOV]',
                 pad='[PAD]'):
        super().__init__()
        if embs is not None:
            embwords, embvecs = embs
            if n_d != len(embvecs[0]):
                logging.warning(
                    f"[WARNINGS] n_d ({n_d}) != word vector size ({len(embvecs[0])}). "
                    f"Use {len(embvecs[0])} for embeddings.")
                n_d = len(embvecs[0])
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oov, self.pad = word2id

    def set_embeddinglayer(self):
        self.embedding = nn.Embedding(self.n_V, self.n_d, padding_idx=)


class WordEmbeddingLayer(nn.Module):

    def __init__(self,
                 word_dim=100,
                 **kwargs):
        super().__init__()



class CharEmbeddingLayer(nn.Module):

    widths = [i for i in range(1, 7+1)]
    dims   = [32, 32, 64, 128, 256, 512, 1024]

    def __init__(self,
                 filters=None,
                 char_dim=50,
                 **kwargs):
        super().__init__()
        if filters is None:
            self.filters = zip(widths, dims)
        else:
            self.filters = filters

    def unicode_():
        pass
