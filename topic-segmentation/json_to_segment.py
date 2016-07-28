#!/usr/bin/env python
# coding: utf-8
import argparse
import codecs
import json
import random

from json_core import Segment, append_to_jsonl


def json_to_segment(json_file, jsonl, random_=False, even=False, empty=False):
    json_object = json.load(json_file)
    topics = json_object['topics']
    gap_count = len(json_object['sents']) - 1
    if random_ or even:
        boundary_count = len(topics) - 1
        boundary_indices = set()
        if random_:
            last_gap_index = gap_count - 1
            while len(boundary_indices) < boundary_count:
                boundary_index = random.randint(0, last_gap_index)
                if boundary_index not in boundary_indices:
                    boundary_indices.add(boundary_index)
            boundary_indices = sorted(boundary_indices)
        elif even:
            step = (gap_count + 1) / (boundary_count + 1)
            boundary_indices = range(1, boundary_count+1)
            boundary_indices = map(lambda x: x * step, boundary_indices)
    elif empty:
        boundary_indices = []
    else:
        boundary_indices = map(lambda topic: topic['end'], topics[:-1])
    segment = Segment(json_object['id'], boundary_indices, gap_count)
    append_to_jsonl(segment, jsonl)


def main(opt):
    with codecs.open(opt.jsonl_file, mode='w', encoding='utf-8') as output:
        for json_path in opt.json_files:
            with codecs.open(json_path, encoding='utf-8') as f:
                json_to_segment(f, output, random_=opt.random, even=opt.even,
                                empty=opt.empty)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--random', action='store_true')
    arg_parser.add_argument('--even', action='store_true')
    arg_parser.add_argument('--empty', action='store_true')
    arg_parser.add_argument('json_files', nargs='+')
    arg_parser.add_argument('jsonl_file')
    opt = arg_parser.parse_args()
    main(opt)
