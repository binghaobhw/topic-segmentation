#!/usr/bin/env python
# coding: utf-8
import argparse
import codecs
import os
import sys

from resource_load import load_json


SEGMENT_SEPARATOR = u'==========\n'


def top_level_to_choi_format(topics, sents, output):
    output.write(SEGMENT_SEPARATOR)
    for topic in topics:
        start = topic['start']
        end = topic['end']
        for i in xrange(start, end + 1):
            sent = sents[i]['text'].lower()
            words = sent.split(' ')
            output.write('{}\n'.format(' '.join(words)))
        output.write(SEGMENT_SEPARATOR)


def get_output_path(out_dir, json_path):
    json_filename = os.path.split(json_path)[1]
    output_filename = '{}.choi'.format(json_filename[:json_filename.rfind('.')])
    output_path = os.path.join(out_dir, output_filename)
    return output_path


def main(opt):
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    output_paths = map(lambda x: get_output_path(opt.out_dir, x),
                       opt.json_paths)
    for json_path, output_path in zip(opt.json_paths, output_paths):
        with codecs.open(json_path, encoding='utf-8') as json_file:
            json_object = load_json(json_file)
            topics = json_object['topics']
            sents = json_object['sents']
        with codecs.open(output_path, mode='w', encoding='utf-8') as output:
            top_level_to_choi_format(topics, sents, output)
    sys.exit(0)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--json_paths', nargs='+')
    arg_parser.add_argument('--out_dir')
    opt = arg_parser.parse_args()
    main(opt)
