import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(levelname)s: %(message)s'
)


if __name__ == '__main__':
    print(5)
    cmd = argparse.ArgumentParser('The testing comopnents of')
    # cmd.add_argument('--gpu', default=1, type=int,
    #                  help='use id of gpu, -1 if cpu')
    # cmd.add_argument('--input_format', default='plain',
    #                  choices=('plain', 'conll', 'conll_char', 'conll_char_vi'),
    #                  help='the input format.')
    cmd.add_argument('--mode', default='train', type=str,
                     choices=('train', 'test'))
    args = cmd.parse_args(sys.argv[2:])
    if args.mode == 'train':
        print('train!')
    else:
        print('test!')
