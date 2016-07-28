# coding: utf-8
import argparse
import codecs


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
            prog='prog',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--vocab_path')
    arg_parser.add_argument('--tab', action='store_true', help='tab as delimiter')
    arg_parser.add_argument('--word_vec_path')
    arg_parser.add_argument('--no_header', action='store_true', help='no header line in word_vec file')
    arg_parser.add_argument('--out_path')
    args = arg_parser.parse_args()
    vocab = set()
    with codecs.open(args.vocab_path, encoding='utf-8') as f:
        for line in f:
            if args.tab:
                word = line.split('\t', 1)[0]
            else:
                word = line.strip()
            vocab.add(word)
    with codecs.open(args.word_vec_path, encoding='utf-8') as f, \
            codecs.open(args.out_path, mode='w', encoding='utf-8') as o:
        if not args.no_header:
            f.readline()
        for line in f:
            word = line.split(' ', 1)[0]
            if word in vocab:
                o.write(line)
