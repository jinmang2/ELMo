import argparse
import logging
import sys

import torch

from .utils import load_config, read_dictionary
from .sample_data.sample_sentences import sents
from .modules.data_handling import (read_corpus,
                                    get_lexicon,
                                    create_batches,
                                    EmbeddingLayer)
from .modules.char_cnn import ConvTokenEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def main(use_cuda=False):
    print('Hello!')
    config = load_config()
    dataset = read_corpus(sents)
    char_fpi = read_dictionary(is_char=True)
    word_fpi = read_dictionary(is_char=False)
    char_lexicon, char_emb_layer = get_lexicon(
        config['token_embedder']['char_dim'], char_fpi, use_cuda)
    word_lexicon, word_emb_layer = get_lexicon(
        config['token_embedder']['word_dim'], word_fpi, use_cuda)
    b_w, b_c, b_l, b_m, recover_ind = create_batches(
        dataset, word_lexicon, char_lexicon, config,
        batch_size=64, shuffle=True, use_cuda=use_cuda)
    token_embedder = ConvTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)
    for w, c, m in zip(b_w, b_c, b_m):
        token_embedding = token_embedder(w, c, m[0].size())
        print(token_embedding)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing comopnents of')
    cmd.add_argument('--gpu', default=1, type=int,
                     help='use id of gpu, -1 if cpu')
    cmd.add_argument('--input_format', default='plain',
                     choices=('plain', 'conll', 'conll_char', 'conll_char_vi'),
                     help='the input format.')
    args = cmd.parse_args(sys.argv[2:])
    if args.gpu != -1:
        if torch.cuda.is_available():
            use_cuda = True
        else:
            raise Exception("GPU 사용 불가")
            use_cuda = False
    else:
        use_cuda = False
    main(use_cuda)
