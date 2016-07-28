# coding: utf-8

import codecs
from sklearn.pipeline import Pipeline
import argparse
import logging
import os

from resource import load_all_data, load_vocab
from feature import Lda, BagOfWords


def build_id2word(vocab):
    return dict([(id_, w) for w, id_ in vocab.iteritems()])


def dump_lda_vec(docs, lda_vecs, sent_set, f):
    for doc, lda_vecs_a_doc in zip(docs, lda_vecs):
        for sent, lda_vec in zip(doc, lda_vecs_a_doc):
            sent_text = ' '.join(sent)
            if sent_text in sent_set:
                continue
            sent_set.add(sent_text)
            lda_vec_text = ' '.join(unicode(i) for i in lda_vec)
            f.write('{}\t{}\n'.format(sent_text, lda_vec_text))


def main(opt):
    with codecs.open(opt.vocab, encoding='utf-8') as f:
        vocab = load_vocab(f)
    id2word = build_id2word(vocab)
    _, docs_train, _ = load_all_data(opt.train_jsons)
    lda = Pipeline([
        ('bow', BagOfWords(vocab=vocab)),
        ('lda', Lda(id2word=id2word, num_topics=opt.num_topics))])
    lda_vec_train = lda.fit_transform(docs_train)

    sent_set = set()
    tmp_path = opt.lda_vec_path + '.tmp'
    with codecs.open(tmp_path, encoding='utf-8', mode='w') as f:
        dump_lda_vec(docs_train, lda_vec_train, sent_set, f)

    if opt.test_jsons:
        _, docs_test, _ = load_all_data(opt.test_jsons)
        lda_vec_test = lda.transform(docs_test)
        with codecs.open(tmp_path, encoding='utf-8', mode='a') as f:
            dump_lda_vec(docs_test, lda_vec_test, sent_set, f)

    with codecs.open(tmp_path, encoding='utf-8') as fin, \
            codecs.open(opt.lda_vec_path, encoding='utf-8', mode='w') as fout:
        fout.write('{} {}\n'.format(len(sent_set), opt.num_topics))
        for line in fin:
            fout.write(line)

    os.remove(tmp_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--vocab')
    arg_parser.add_argument('--num_topics', type=int)

    arg_parser.add_argument('--train_jsons', nargs='+')
    arg_parser.add_argument('--test_jsons', nargs='*')
    arg_parser.add_argument('--lda_vec_path')

    opt = arg_parser.parse_args()

    main(opt)
