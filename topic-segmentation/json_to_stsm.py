#!/usr/bin/env python
# coding: utf-8
import argparse
import codecs
import os
import sys

from resource_load import load_json


LINE_FORMAT = u'{}\n'


def get_index(name, names, name_dict):
    if name not in name_dict:
        index = len(names)
        name_dict[name] = index
        names.append(name)
    else:
        index = name_dict[name]
    return index


def bag_of_words_for_sent(sent, vocab, inverted_vocab):
    """
    Parameter
    ---------
    sent : str
    vocab : {str: int}
    inverted_vocab : {int: str}
    """
    word_index_count_dict = {}
    for word in sent.split(' '):
        if not word:
            continue
        index = get_index(word, vocab, inverted_vocab)
        if word not in word_index_count_dict:
            word_index_count_dict[word] = {'index': index, 'count': 0}
        word_index_count_dict[word]['count'] += 1
    return word_index_count_dict.values()


def top_level_to_stsm(topics, sents, meta, data, seg, vocab, inverted_vocab):
    boundaries_a_doc = []
    line_count = 0
    for topic in topics:
        topic_line_count = 0
        start = topic['start']
        end = topic['end']
        for i in xrange(start, end + 1):
            sent_text = sents[i]['text'].lower()
            word_index_count_list = bag_of_words_for_sent(
                    sent_text, vocab, inverted_vocab)
            data.append(word_index_count_list)
            boundaries_a_doc.append(0)
            topic_line_count += 1

        boundaries_a_doc[-1] = 1
        line_count += topic_line_count
    meta.append(line_count)
    seg.append(boundaries_a_doc)


def main(opt):
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    meta_path, data_path, seg_path, vocab_path, name_path = map(
        lambda x: os.path.join(opt.out_dir, opt.out_prefix + x),
        ['.meta', '.data', '.seg', '.vocab', '.name'])
    meta = []
    data = []
    seg = []
    vocab = []
    inverted_vocab = {}
    names = []
    for json_path in opt.json_paths:
        with codecs.open(json_path, encoding='utf-8') as json_file:
            json_object = load_json(json_file)
            topics = json_object['topics']
            sents = json_object['sents']
            meeting_name = json_object['id']
            names.append(meeting_name)
            top_level_to_stsm(topics, sents, meta, data, seg, vocab,
                              inverted_vocab)
    with codecs.open(meta_path, mode='w', encoding='utf-8') as meta_file:
        for i_meta in meta:
            meta_file.write(LINE_FORMAT.format(i_meta))
    with codecs.open(data_path, mode='w', encoding='utf-8') as data_file:
        for i_data in data:
            data_file.write(LINE_FORMAT.format(','.join(
                map(lambda x: '{}:{}'.format(x['index'], x['count']), i_data))))
    with codecs.open(seg_path, mode='w', encoding='utf-8') as seg_file:
        for i_seg in seg:
            seg_file.write(LINE_FORMAT.format(','.join(map(unicode, i_seg))))
    with codecs.open(vocab_path, mode='w', encoding='utf-8') as vocab_file:
        for i_vocab in vocab:
            vocab_file.write(LINE_FORMAT.format(i_vocab))
    with codecs.open(name_path, mode='w', encoding='utf-8') as name_file:
        for name in names:
            name_file.write(LINE_FORMAT.format(name))
    sys.exit(0)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--json_paths', nargs='+')
    arg_parser.add_argument('--out_dir')
    arg_parser.add_argument('--out_prefix')
    opt = arg_parser.parse_args()
    main(opt)
