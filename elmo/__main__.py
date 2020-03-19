import argparse
import sys
from .utils import load_config

def mm():
    print('Hello!')

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing comopnents of')
    cmd.add_argument('--gpu', default=1, type=int, help='use id of gpu, -1 if cpu')
    cmd.add_argument('--input_format', default='plain', choices=('plain', 'conll', 'conll_char', 'conll_char_vi'),
                     help='the input format.')
    args = cmd.parse_args(sys.argv[2:])
    print(args.gpu)
    print(args.input_format)
    config = load_config()
    print(config)
    mm()
