#!/usr/bin/env python
# coding: utf-8
import argparse
import codecs
import json
import os

from json_core import Topic, to_json


def segment_to_topic(segments):
    start = 0
    topics = []
    for end in segments:
        topics.append(Topic(start, end, u'', []))
        start = end + 1
    return topics


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('segment')
    arg_parser.add_argument('json_dir')
    arg_parser.add_argument('out_dir')
    args = arg_parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    with codecs.open(args.segment, encoding='utf-8') as segment_file:
        for line in segment_file:
            line = line.strip()
            name, num_lines, num_segments, segments = line.split(u' ', 3)
            num_lines = int(num_lines)
            num_segments = int(num_segments)
            segments = map(int, segments.split(u','))
            topics = segment_to_topic(segments)
            json_path = os.path.join(args.json_dir, name + '.json')
            json_object = json.load(codecs.open(json_path, encoding='utf-8'))
            lines = json_object['lines']
            out_path = os.path.join(args.out_dir, name + '.json')
            to_json(lines, topics, codecs.open(out_path, encoding='utf-8', mode='wb'))

