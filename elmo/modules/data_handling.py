import logging

import torch.nn as nn

def add_begin_end_sentence_token(sents, max_chars=50):
    dataset, textset = [], []
    for sent in sents:
        # Add begin of sentence(bos)
        data = ['<bos>']
        text = []
        for token in sent:
            text.append(token)
            # ELMo's input is character
            # Since ElMo uses char-CNN, input_dim must be SAME
            # if numChars+2 < max_chars: why +2? bos & eos
            #     pad values to pad_id
            # else:
            #     cut token:= token[:max_chars - 2]
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        # Add end of sentence(eos)
        data.append('<eos>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset

class EmbeddingLayer(nn.Module):
    """
    EmbeddingLayer

    두 가지 역할을 수행
    1. word/character를 사전 규칙에 따라 index로 변환
    2. config['token_embedder']['char_dim']으로 차원을 축소
    """
    def __init__(self, n_d, word2id, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
        super(EmbeddingLayer, self).__init__()
        if embs is not None:
            embwords, embvecs = embs
            logging.info(f"{len(word2id)} pre-trained word embeddings loaded.")
            if n_d != len(embvecs[0]):
                logging.warning(f"[WARNINGS] n_d ({n_d}) != word vector size ({len(embvecs[0])}). "
                                f"Use {len(embvecs[0])} for embeddings.")
                n_d = len(embvecs[0])
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        # n_V -> n_d, 차원 축소
        self.embedding = nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if embs is not None:
            weight = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            logging.info("embedding shape: {}".format(weight.size()))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input_):
        return self.embedding(input_)
