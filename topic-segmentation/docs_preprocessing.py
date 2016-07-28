# coding: utf-8

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import argparse
import codecs
import os

from json_core import to_json
from resource_load import load_json, load_set

EMPTY_TOKEN = '<empty>'


def tokenize(docs):
    """
    Args:
        docs ([[str]])
    """
    return map(lambda doc: map(lambda sent: nltk.word_tokenize(sent), doc), docs)


def pos_tag(docs):
    """
    Args:
        docs ([[[str]]])
    """
    return map(lambda doc: nltk.pos_tag_sents(doc), docs)

lemmatizer = WordNetLemmatizer()


def lemmatize_sent(sent):
    """
    Args:
        sent ([(str, str)]): pos tag for sent

    Returns:
        [str]
    """
    def treebank_to_wordnet(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    return map(lambda t: lemmatizer.lemmatize(t[0], pos=treebank_to_wordnet(t[1])), sent)


def lemmatize(docs):
    """
    Parameter
    ---------
    docs : [[[(str, str)]]]
        pos tags for docs

    Returns
    -------
    out : [[[str]]]
    """
    def lemmatize_doc(doc):
        return map(lemmatize_sent, doc)

    return map(lemmatize_doc, docs)


def remove_stopwords(sent, stopwords, use_empty_token=False):
    filtered = filter(lambda w: w not in stopwords, sent)
    if len(filtered) == 0 and use_empty_token:
        filtered.append(EMPTY_TOKEN)
    return filtered


def preprocess(docs, lowercase=False, pos_tagging=False, lemmatization=False,
               stopwords=set(), use_empty_token=False):
    """
    Parameter
    ---------
    docs : [[str]]

    Returns
    -------
    out : [[[str]]]
    """
    if lowercase:
        docs = map(lambda doc: map(lambda sent: sent.lower(), doc), docs)
    x = tokenize(docs)
    if pos_tagging or lemmatization:
        x = pos_tag(x)
    if lemmatization:
        x = lemmatize(x)
    output = []
    for doc in x:
        o_doc = map(lambda s: remove_stopwords(s, stopwords, use_empty_token=use_empty_token), doc)
        output.append(o_doc)
    return output


def main(opt):
    if not os.path.exists(opt.out_dir):
        os.mkdir(opt.out_dir)

    if opt.stopwords:
        with codecs.open(opt.stopwords, encoding='utf-8') as f:
            stopwords = load_set(f)
    else:
        stopwords = set()

    for json_path in opt.json_paths:
        out_path = os.path.join(opt.out_dir, os.path.basename(json_path))
        with codecs.open(out_path, mode='w', encoding='utf-8') as out_file, \
                codecs.open(json_path, encoding='utf-8') as in_file:
            doc = load_json(in_file)
            sents = doc['sents']
            sents_text = map(lambda s: s['text'], sents)
            result = preprocess(
                    [sents_text], lowercase=True, pos_tagging=False,
                    lemmatization=True, stopwords=stopwords, use_empty_token=opt.use_empty_token)[0]

            sents_text = map(lambda s: ' '.join(s), result)
            for sent, sent_text in zip(sents, sents_text):
                sent['text'] = sent_text
            to_json(doc['id'], sents, doc['topics'], out_file)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
            prog='prog',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--json_paths', nargs='+')
    arg_parser.add_argument('--out_dir')
    arg_parser.add_argument('--stopwords', default=None)
    arg_parser.add_argument('--use_empty_token', action='store_true')
    opt = arg_parser.parse_args()
    main(opt)
