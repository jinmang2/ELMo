# -*- coding: utf-8 -*-

# PSL: Python Standard Library
import collections
import logging
import codecs
import random
import json
import os

# Python Installed Library
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# User Defined Library
from .modules.elmo import ElmobiLm
from .modules.lstm import LstmbiLm
from .modules.token_embedder import ConvTokenEmbedder, LstmTokenEmbedder


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')


def dict2namedtuple(dic):
    return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_list(sents, max_chars=None):
    dataset, textset = [], []
    for sent in sents:
        data = ['<bos>']
        text = []
        for token in sent:
            text.append(token)
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset


def recover(li, ind):
    dummy = list(range(len(ind)))
    dummy.sort(key=lambda l: ind[l])
    li = [li[i] for i in dummy]
    return li


# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=False, sort=True, text=None):
    oov = '<oov>'
    pad = '<pad>'

    ind = list(range(len(x)))
    lst = perm or ind
    if shuffle:
        random.shuffle(lst)
    if sort:
        lst.sort(key=lambda l: -len(x[l]))
        
    x = [x[i] for i in lst]
    ind = [ind[i] for i in lst]
    if text is not None:
        text = [text[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks, batches_text, batches_ind = [], [], [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        """ -------- Create one_batch START -------- """
        x_b = x[start_id: end_id]
        batch_size = len(x_b)
        lst = list(range(batch_size))
        if sort:
            lst.sort(key=lambda l: -len(x[l]))
        # shuffle the sentences by
        x_b = [x_b[i] for i in lst]
        lens = [len(x_b[i]) for i in lst]
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
            ]
            assert ((bow_id is not None) and 
                    (eow_id is not None) and
                    (oov_id is not None) and
                    (pad_id is not None))
            if config['token_embedder']['name'].lower() == 'cnn':
                max_chars = config['token_embedder']['max_characters_per_token']
                assert max([len(w) for i in lst for w in x_b[i]]) + 2 <= max_chars
            elif config['token_embedder']['name'].lower() == 'lstm':
                max_chars = max([len(w) for i in lst for w in x_b[i]]) + 2
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
        """ -------- Create one_batch END -------- """
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)
        batches_ind.append(ind[start_id: end_id])
        if text is not None:
            batches_text.append(text[start_id: end_id])

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        batches_masks = [batches_masks[i] for i in perm]
        batches_ind = [batches_ind[i] for i in perm]
        if text is not None:
            batches_text = [batches_text[i] for i in perm]

    logging.info("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len / len(x)))
    recover_ind = [item for sublist in batches_ind for item in sublist]
    if text is not None:
        return batches_w, batches_c, batches_lens, batches_masks, batches_text, recover_ind
    return batches_w, batches_c, batches_lens, batches_masks, recover_ind

class Model(nn.Module):
    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.config = config
        
        if config['token_embedder']['name'].lower() == 'cnn':
          self.token_embedder = ConvTokenEmbedder(
            config, word_emb_layer, char_emb_layer, use_cuda)
        elif config['token_embedder']['name'].lower() == 'lstm':
          self.token_embedder = LstmTokenEmbedder(
            config, word_emb_layer, char_emb_layer, use_cuda)

        if config['encoder']['name'].lower() == 'elmo':
          self.encoder = ElmobiLm(config, use_cuda)
        elif config['encoder']['name'].lower() == 'lstm':
          self.encoder = LstmbiLm(config, use_cuda)

        self.output_dim = config['encoder']['projection_dim']
    
    def forward(self, word_inp, chars_package, mask_package):
        token_embedding = self.token_embedder(word_inp, chars_package,
            (mask_package[0].size(0), mask_package[0].size(1))
        if self.config['encoder']['name'] == 'elmo':
            mask = Variable(mask_package[0])
            if self.use_cuda:
                mask = mask.cuda()
            encoder_output = self.encoder(token_embedding, mask)
            sz = encoder_output.size()
            token_embedding = torch.cat(
                [token_embedding, token_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
            encoder_output = torch.cat(
                [token_embedding, encoder_output], dim=0)
        elif self.config['encoder']['name'] == 'lstm':
            encoder_output = self.encoder(token_embedding)
        else:
            raise ValueError(f'Unknown encoder: {self.config['encoder']['name']}')
        return encoder_output
        
    def load_model(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'),
                                                       map_location=lambda storage, loc: storage))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'),
                                                map_location=lambda storage, loc: storage))
        
        
class EmbeddingLayer(nn.Module):
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


class Embedder(object):
    def __init__(self, model_dir, batch_size=64):
        self.model_dir = model_dir
        self.model, self.config = self.get_model()
        self.batch_size = batch_size

    def get_model(self):
        # torch.cuda.set_device(1)
        self.use_cuda = torch.cuda.is_available()
        # load the model configurations
        args2 = dict2namedtuple(json.load(codecs.open(
            os.path.join(self.model_dir, 'config.json'), 'r', encoding='utf-8')))

        with open(os.path.join(self.model_dir, args2.config_path), 'r') as fin:
            config = json.load(fin)

        # For the model trained with character-based word encoder.
        if config['token_embedder']['char_dim'] > 0:
            self.char_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'char.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.char_lexicon[token] = int(i)
            char_emb_layer = EmbeddingLayer(
                config['token_embedder']['char_dim'], self.char_lexicon, fix_emb=False, embs=None)
            logging.info('char embedding size: ' +
                        str(len(char_emb_layer.word2id)))
        else:
            self.char_lexicon = None
            char_emb_layer = None

        # For the model trained with word form word encoder.
        if config['token_embedder']['word_dim'] > 0:
            self.word_lexicon = {}
            with codecs.open(os.path.join(self.model_dir, 'word.dic'), 'r', encoding='utf-8') as fpi:
                for line in fpi:
                    tokens = line.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, i = tokens
                    self.word_lexicon[token] = int(i)
            word_emb_layer = EmbeddingLayer(
                config['token_embedder']['word_dim'], self.word_lexicon, fix_emb=False, embs=None)
            logging.info('word embedding size: ' +
                        str(len(word_emb_layer.word2id)))
        else:
            self.word_lexicon = None
            word_emb_layer = None

        # instantiate the model
        model = Model(config, word_emb_layer, char_emb_layer, self.use_cuda)

        if self.use_cuda:
            model.cuda()

        logging.info(str(model))
        model.load_model(self.model_dir)

        # read test data according to input format

        # configure the model to evaluation mode.
        model.eval()
        return model, config

    def sents2elmo(self, sents, output_layer=-1):
        read_function = read_list

        if self.config['token_embedder']['name'].lower() == 'cnn':
            test, text = read_function(sents, self.config['token_embedder']['max_characters_per_token'])
        else:
            test, text = read_function(sents)

        # create test batches from the input data.
        test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
            test, self.batch_size, self.word_lexicon, self.char_lexicon, self.config, text=text)

        cnt = 0

        after_elmo = []
        for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
            output = self.model.forward(w, c, masks)
            for i, text in enumerate(texts):

                if self.config['encoder']['name'].lower() == 'lstm':
                    data = output[i, 1:lens[i]-1, :].data
                    if self.use_cuda:
                        data = data.cpu()
                    data = data.numpy()
                elif self.config['encoder']['name'].lower() == 'elmo':
                    data = output[:, i, 1:lens[i]-1, :].data
                    if self.use_cuda:
                        data = data.cpu()
                    data = data.numpy()

                if output_layer == -1:
                    payload = np.average(data, axis=0)
                elif output_layer == -2:
                    payload = data
                else:
                    payload = data[output_layer]
                after_elmo.append(payload)

                cnt += 1
                if cnt % 1000 == 0:
                    logging.info('Finished {0} sentences.'.format(cnt))

        after_elmo = recover(after_elmo, recover_ind)
        return after_elmo
