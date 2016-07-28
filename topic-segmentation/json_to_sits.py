#!/usr/bin/env python
# coding: utf-8
import argparse
import codecs
import os
import sys

from resource_load import load_json

LINE_PATTERN = u'{}\n'
WORDS_LINE_PATTERN = u'{}\t{}\n'
INFO_LINE_PATTERN = u'{}\t{}\t{}\n'
NEW_LINE = u'\n'


def get_index(name, names, name_dict):
    if name not in name_dict:
        index = len(names)
        name_dict[name] = index
        names.append(name)
    else:
        index = name_dict[name]
    return index


def get_sent_speaker_index(speaker, speakers, speaker_dict):
    return get_index(speaker, speakers, speaker_dict)


def get_sent_word_indices(sent_text, words, word_dict):
    splits = sent_text.split(' ')
    return map(lambda word: get_index(word, words, word_dict), splits)


def doc_to_sits(sents, topics, words, word_dict, speakers, speaker_dict,
                segment_nums, word_indices, speaker_indices, infos):
    doc_word_indices = []
    doc_speaker_indices = []
    doc_infos = []
    for i, sent in enumerate(sents):
        speaker = sent['speaker']
        sent_text = sent['text'].lower()
        doc_infos.append((i, speaker, sent_text))
        doc_word_indices.append(
            get_sent_word_indices(sent_text, words, word_dict))
        doc_speaker_indices.append(
            get_sent_speaker_index(speaker, speakers, speaker_dict))
    segment_nums.append(len(topics))
    word_indices.append(doc_word_indices)
    speaker_indices.append(doc_speaker_indices)
    infos.append(doc_infos)


def output_words(out, word_indices):
    num_docs = len(word_indices)
    num_sents = sum(map(len, word_indices)) + num_docs
    out.write(LINE_PATTERN.format(num_docs))
    out.write(LINE_PATTERN.format(num_sents))
    for doc_word_indices in word_indices:
        for sent_word_indices in doc_word_indices:
            out.write(WORDS_LINE_PATTERN.format(len(sent_word_indices),
                                                ' '.join(map(unicode,
                                                             sent_word_indices))))
        out.write(NEW_LINE)


def output_shows(out, names):
    for doc_name, num_sents in names:
        for i in xrange(num_sents):
            out.write(LINE_PATTERN.format(doc_name))
        out.write(NEW_LINE)


def output_authors(out, speaker_indices):
    for doc_speaker_indices in speaker_indices:
        for sent_speaker_index in doc_speaker_indices:
            out.write(LINE_PATTERN.format(sent_speaker_index))
        out.write(LINE_PATTERN.format(-1))


def output_voc(out, words):
    for word in words:
        out.write(LINE_PATTERN.format(word))


def output_whois(out, speakers):
    for speaker in speakers:
        out.write(LINE_PATTERN.format(speaker))


def output_text(out, infos):
    for doc_infos in infos:
        for sent_info in doc_infos:
            out.write(INFO_LINE_PATTERN.format(*sent_info))
        out.write(NEW_LINE)


def output_segment(out, segment_nums):
    for segment_num in segment_nums:
        out.write(LINE_PATTERN.format(segment_num))


def main(opt):
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)
    words_path, shows_path, authors_path, voc_path, whois_path, text_path, segment_path = map(
        lambda x: os.path.join(opt.out_dir, opt.out_prefix + x),
        ['.words', '.shows', '.authors', '.voc', '.whois', '.text', '.segment'])
    word_indices = []
    names = []
    speaker_indices = []
    words = []
    word_dict = {}
    speakers = []
    speaker_dict = {}
    infos = []
    segment_nums = []
    for json_path in opt.json_paths:
        with codecs.open(json_path, encoding='utf-8') as json_file:
            json_object = load_json(json_file)
            doc_name = json_object['id']
            sents = json_object['sents']
            names.append((doc_name, len(sents)))
            topics = json_object['topics']
            doc_to_sits(sents, topics, words, word_dict, speakers, speaker_dict,
                        segment_nums, word_indices, speaker_indices, infos)
    with codecs.open(words_path, mode='w', encoding='utf-8') as out:
        output_words(out, word_indices)
    with codecs.open(shows_path, mode='w', encoding='utf-8') as out:
        output_shows(out, names)
    with codecs.open(authors_path, mode='w', encoding='utf-8') as out:
        output_authors(out, speaker_indices)
    with codecs.open(text_path, mode='w', encoding='utf-8') as out:
        output_text(out, infos)
    with codecs.open(voc_path, mode='w', encoding='utf-8') as out:
        output_voc(out, words)
    with codecs.open(whois_path, mode='w', encoding='utf-8') as out:
        output_whois(out, speakers)
    with codecs.open(segment_path, mode='w', encoding='utf-8') as out:
        output_segment(out, segment_nums)
    sys.exit(0)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--json_paths', nargs='+')
    arg_parser.add_argument('--out_dir')
    arg_parser.add_argument('--out_prefix')
    opt = arg_parser.parse_args()
    main(opt)
