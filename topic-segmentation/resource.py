# coding: utf-8

import json
import codecs
import numpy as np
import argparse

EMPTY_TOKEN = '<empty>'


def load_list(f):
    return [line.strip() for line in f]


def load_set(f):
    set_ = set()
    for line in f:
        set_.add(line.strip())
    return set_


def load_json(f):
    return json.load(f)


def load_data(json_file):
    """
    Parameter
    ---------
    json_file : readable

    Returns
    -------
    id_ : str
    doc : [[str]]
        doc[i][j] is j-th word in i-th sent
    y : np.ndarray
        shape (doc_len - 1, )
        y[i] is label for i-th gap
    """
    json_object = json.load(json_file)
    id_ = json_object['id']

    sents = json_object['sents']
    doc = map(lambda sent: sent['text'].split(' '), sents)

    y = np.full(len(sents) - 1, -1, dtype=np.int)
    topics = json_object['topics']
    boundary_indices = map(lambda t: t['end'], topics[:-1])
    y[boundary_indices] = 1

    return id_, doc, y


def load_all_data(json_paths):
    """
    Parameter
    ---------
    paths : [str]

    Returns
    -------
    ids : [str]
    docs : [[[str]]]
        doc[i][j][k] is k-th word of j-th sent in i-th doc
    ys : [np.ndarray]
        ys[i][j] is label for j-th gap in i-th doc
    """
    ids = []
    docs = []
    ys = []
    for json_path in json_paths:
        with codecs.open(json_path, encoding='utf-8') as f:
            id_, doc, y = load_data(f)
            ids.append(id_)
            docs.append(doc)
            ys.append(y)
    return ids, docs, ys


def build_vocab(docs, stopwords):
    """
    Parameters
    ----------
    docs : [[[str]]]
    stopwords : set

    Returns
    -------
    vocab : {str: int}
    """
    vocab = {}
    for doc in docs:
        for sent in doc:
            filtered = filter(lambda w: w not in stopwords, sent)
            if len(filtered) == 0:
                filtered.append(EMPTY_TOKEN)
            for w in filtered:
                if w not in vocab:
                    vocab[w] = len(vocab)
    return vocab


def dump_vocab(vocab, vocab_file):
    for term, index in vocab.iteritems():
        vocab_file.write(u'{}\t{}\n'.format(term, index))


def load_vocab(vocab_file):
    vocab = {}
    for line in vocab_file:
        line = line.strip()
        term, index = line.split('\t', 1)
        vocab[term] = int(index)
    return vocab


def load_vocab_from_list(f):
    vocab = {}
    for i, line in enumerate(f):
        w = line.strip()
        vocab[w] = i
    return vocab


def load_word_vec(word_vec_file):
    """
    Parameters
    ----------
    word_vec_file : readable
        should contain header which indicates vocab_size and vec_size

    Returns
    -------
    vocab : {str, int}
    word_matrix : np.ndarray
        dtype: np.float32
    """
    vocab = {}
    header = word_vec_file.readline()
    vocab_size, vec_size = map(int, header.split(' ', 1))
    word_matrix = np.zeros((vocab_size, vec_size), dtype=np.float32)
    for i, line in enumerate(word_vec_file):
        parts = line.strip().split(" ")
        if len(parts) != vec_size + 1:
            raise ValueError("invalid vector on line %s" % (i))
        word = parts[0]
        weights = np.array(list(map(float, parts[1:])))
        word_id = len(vocab)
        vocab.setdefault(word, word_id)
        word_matrix[word_id] = weights
    if EMPTY_TOKEN not in vocab:
        raise Exception('No empty token in word_vec_file')
    return vocab, word_matrix


def load_sent_vec(f):
    """
    Parameters
    ----------
    f : readable
        should contain header which indicates vocab_size and vec_size

    Returns
    -------
    vocab : {str, int}
    sent_matrix : np.ndarray
        dtype: np.float32
    """
    vocab = {}
    header = f.readline()
    vocab_size, vec_size = map(int, header.split(' ', 1))
    sent_matrix = np.zeros((vocab_size, vec_size), dtype=np.float32)
    for i, line in enumerate(f):
        sent, vec_part = line.strip().split("\t", 1)
        weights = np.array(list(map(float, vec_part.split(' '))))
        sent_id = len(vocab)
        vocab.setdefault(sent, sent_id)
        sent_matrix[sent_id] = weights
    return vocab, sent_matrix


def main(opt):
    if opt.build_vocab:
        ids, docs, ys = load_all_data(opt.json_paths)
        if opt.stopwords:
            stopwords = load_set(opt.stopwords)
        else:
            stopwords = set()
        vocab = build_vocab(docs, stopwords)
        with codecs.open(opt.vocab_path, encoding='utf-8', mode='w') as f:
            dump_vocab(vocab, f)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--build_vocab', action='store_true')
    arg_parser.add_argument('--stopwords', default=None)
    arg_parser.add_argument('--vocab_path')
    arg_parser.add_argument('--json_paths', nargs='+')

    opt = arg_parser.parse_args()
    main(opt)
