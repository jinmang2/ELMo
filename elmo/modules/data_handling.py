import logging
import random

import torch
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


# For the model trained with character-based word encoder.
def get_lexicon(dim, fpi, use_cuda=False):
    if dim > 0:
        lexicon = {}
        for line in fpi:
            tokens = line.strip().split('\t')
            if len(tokens) == 1:
                tokens.insert(0, '\u3000')
            token, i = tokens
            lexicon[token] = int(i)
        fpi.close()
        emb_layer = EmbeddingLayer(
            dim, lexicon, fix_emb=False, embs=None)
        if use_cuda:
            emb_layer = emb_layer.cuda()
        logging.info('embedding size: ' +
                     str(len(emb_layer.word2id)))
    else:
        lexicon = None
        emb_layer = None
    return lexicon, emb_layer


def create_batches(x,
                   word2id,
                   char2id,
                   config,
                   batch_size=64,
                   shuffle=False,
                   use_cuda=False,
                   oov='<oov>',
                   pad='<pad>'):
    ind = list(range(len(x)))
    if shuffle:
        random.shuffle(ind)
    x = [x[i] for i in ind]

    sum_len       = 0.0
    batches_w     = []
    batches_c     = []
    batches_lens  = []
    batches_masks = []
    batches_text  = []
    batches_ind   = []
    size          = batch_size
    nbatch = (len(x) - 1) // size + 1

    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        # Create one_batch---------------------------------------
        x_b = x[start_id: end_id]
        batch_size = len(x_b)
        lens = [len(x_b[i]) for i in ind]
        max_len = max(lens)

        # get a batch of word id whose size is (batch x max_len)
        if word2id is not None:
            oov_id = word2id.get(oov, None)
            pad_id = word2id.get(pad, None)
            assert oov_id is not None and pad_id is not None
            batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
            for i, x_i in enumerate(x_b):
                for j, x_ij in enumerate(x_i):
                    batch_w[i][j] = word2id.get(x_ij, oov_id)
        else:
            batch_w = None

        # get a batch of character id whose size is (batch x max_chars)
        if char2id is not None:
            bow_id, eow_id, oov_id, pad_id = [
                char2id.get(key, None)
                for key in ('<eow>', '<bow>', oov, pad)
            ] # 왜 거꾸로 받지???ㄷㄷ;;
            assert ((bow_id is not None) and
                    (eow_id is not None) and
                    (oov_id is not None) and
                    (pad_id is not None))
            if config['token_embedder']['name'].lower() == 'cnn':
                max_chars = config['token_embedder']['max_characters_per_token']
                assert max([len(w) for i in ind for w in x_b[i]]) + 2 <= max_chars
            elif config['token_embedder']['name'].lower() == 'lstm':
                max_chars = max([len(w) for i in ind for w in x_b[i]]) + 2
            else:
                raise ValueError('Unknown token_embedder: {0}'.format(config['token_embedder']['name']))
            batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)
            for i, x_i in enumerate(x_b):
                for j, x_ij in enumerate(x_i):
                    batch_c[i][j][0] = bow_id
                    if x_ij in ['<bos>', '<eos>']:
                        batch_c[i][j][1] = char2id.get(x_ij)
                        batch_c[i][j][2] = eow_id
                    else:
                        for k, c in enumerate(x_ij):
                            batch_c[i][j][k+1] = char2id.get(c, oov_id)
                        batch_c[i][j][len(x_ij)+1] = eow_id
        else:
            batch_c = None

        masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

        for i, x_i in enumerate(x_b):
            for j in range(len(x_i)):
                masks[0][i][j] = 1
                if j + 1 < len(x_i):
                    masks[1].append(i * max_len + j)
                if j > 0:
                    masks[2].append(i * max_len + j)

        assert len(masks[1]) <= batch_size * max_len
        assert len(masks[2]) <= batch_size * max_len

        masks[1] = torch.LongTensor(masks[1])
        masks[2] = torch.LongTensor(masks[2])
        # -------------------------------------------------------
        bw, bc, blens, bmasks = batch_w, batch_c, lens, masks
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)
        batches_ind.append(ind[start_id: end_id])

    logging.info("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len / len(x)))
    recover_ind = [item for sublist in batches_ind for item in sublist]
    if use_cuda:
        batches_w[0] = batches_w[0].cuda()
        batches_c[0] = batches_c[0].cuda()
        batches_masks[0] = [mask.cuda() for mask in batches_masks[0]]
    return batches_w, batches_c, batches_lens, batches_masks, recover_ind

class EmbeddingLayer(nn.Module):
    """
    EmbeddingLayer

    두 가지 역할을 수행
    1. word/character를 사전 규칙에 따라 index로 변환
    2. config['token_embedder']['char_dim']으로 차원을 축소
    """
    def __init__(self,
                 n_d,
                 word2id,
                 embs=None,
                 fix_emb=True,
                 oov='<oov>',
                 pad='<pad>',
                 normalize=True):
        super(EmbeddingLayer, self).__init__()
        if embs is not None:
            embwords, embvecs = embs
            logging.info(f"{len(word2id)} pre-trained word embeddings loaded.")
            if n_d != len(embvecs[0]):
                logging.warning(f"[WARNINGS] n_d ({n_d}) != word vector size "
                                f"({len(embvecs[0])}). ",
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
