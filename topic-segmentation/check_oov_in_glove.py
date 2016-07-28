#!/usr/bin/env python
# coding: utf-8
import argparse
import codecs
import json
import os
import sys


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('json_dir')
    arg_parser.add_argument('glove_file')
    args = arg_parser.parse_args()
    train_dir = os.path.join(args.json_dir, 'train')
    test_dir = os.path.join(args.json_dir, 'test')
    json_paths = map(lambda x: os.path.join(train_dir, x),
                     os.listdir(train_dir))
    json_paths.extend(map(lambda x: os.path.join(test_dir, x),
                     os.listdir(test_dir)))
    vob = set()
    with codecs.open(args.glove_file, encoding='utf-8') as glove_file:
        for line in glove_file:
            line = line.strip()
            if line:
                vob.add(line.split(' ', 1)[0])
    oov = set()
    in_vob = set()
    for json_path in json_paths:
        with codecs.open(json_path, encoding='utf-8') as json_file:
            json_object = json.load(json_file)
            lines = json_object['lines']
            for i, line in enumerate(lines):
                words = line['text']
                empty_line = True
                for word in words:
                    word = word.lower()
                    if word in vob:
                        in_vob.add(word)
                        if empty_line:
                            empty_line = False
                    else:
                        oov.add(word)
                if empty_line:
                    print '{}: line {}'.format(json_path, i)
    for w in oov:
        print w
    with codecs.open('word-in-glove.txt', encoding='utf-8', mode='w') as f:
        for w in in_vob:
            f.write('{}\n'.format(w))

