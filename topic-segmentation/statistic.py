#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import argparse
import codecs
import json
import operator
from resource import load_set


def count(json_files):
    word_count = 0
    sent_count = 0
    topic_count = 0
    vocab = set()
    for json_file in json_files:
        json_ = json.load(json_file)
        sent_count += json_['sentCount']
        topic_count += json_['topicCount']
        for sent in json_['sents']:
            words = sent['text'].split()
            word_count += len(words)
            for w in words:
                vocab.add(w)
    print('vocab_size: {}, word_count: {}, sent_count: {}, topic_count: {}'.format(
        len(vocab), word_count, sent_count, topic_count))


def top_bigrams_in_new_start_sent(json_files):
    bigram_count = {}
    for json_file in json_files:
        doc = json.load(json_file)
        sents = doc['sents']
        new_start_sents = map(lambda t: sents[t['start']], doc['topics'][1:])
        for sent in new_start_sents:
            words = sent['text'].split()
            for i in xrange(len(words) - 1):
                bigram = ' '.join(words[i:i+2])
                if bigram not in bigram_count:
                    bigram_count[bigram] = 1
                else:
                    bigram_count[bigram] += 1
    s = sorted(bigram_count.items(), key=operator.itemgetter(1), reverse=True)
    for b, c in s:
        if c == 1:
            break
        print('{} : {}'.format(b, c))


def top_words_in_new_start_sent(json_files, opt):
    if opt.stopwords:
        with codecs.open(opt.stopwords, encoding='utf-8') as f:
            stopwords = load_set(f)
    else:
        stopwords = set()

    word_count = {}
    for json_file in json_files:
        doc = json.load(json_file)
        sents = doc['sents']
        new_start_sents = map(lambda t: sents[t['start']], doc['topics'][1:])
        for sent in new_start_sents:
            words = sent['text'].split()
            for w in words:
                if w in stopwords:
                    continue
                if w not in word_count:
                    word_count[w] = 1
                else:
                    word_count[w] += 1
    s = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    for w, c in s:
        if c == 1:
            break
        print('{} : {}'.format(w, c))


def main(opt):
    json_files = map(lambda p: codecs.open(p, encoding='utf-8'), opt.json_files)
    if opt.count:
        count(json_files)
    elif opt.top_words_in_start:
        top_words_in_new_start_sent(json_files, opt)
    elif opt.top_bigrams_in_start:
        top_bigrams_in_new_start_sent(json_files)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--count', action='store_true')
    arg_parser.add_argument('--top_words_in_start', action='store_true')
    arg_parser.add_argument('--top_bigrams_in_start', action='store_true')
    arg_parser.add_argument('--stopwords', default=None)
    arg_parser.add_argument('json_files', nargs='+')
    opt = arg_parser.parse_args()
    main(opt)
