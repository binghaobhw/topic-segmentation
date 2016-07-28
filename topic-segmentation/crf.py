# coding: utf-8

import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn_crfsuite import CRF
import argparse
import sys
import logging
import yaml
from collections import Counter

from evaluation import average_window_diff, window_diff_scorer, \
        average_pk
from resource import load_all_data, load_set, load_list, load_vocab, \
        load_sent_vec
from feature import CueWord, CueBigram, DepthScore, \
        BagOfWords, EstimatorWrapper, LdaLookup, \
        CrfFeatureTransformer, LexicalFeature
from json_core import pred_to_segment_jsonl


logger = logging.getLogger(__name__)

np.random.seed = 721


def build_id2word(vocab):
    return dict([(id_, w) for w, id_ in vocab.iteritems()])


def param_search(model, feature_names, X, y):
    scorer = window_diff_scorer(punish_zero_seg=True, boundary_label='1')
    k_fold = KFold(len(y), n_folds=3, shuffle=True, random_state=721)
    param_space = {
        'features__window_size': np.linspace(5, 10, num=2, dtype=np.int)}
    for feature_name in feature_names:
        if feature_name == 'depth_score':
            param_space['features__depth_score__depth_score__block_size'] = np.linspace(15, 25, num=2, dtype=np.int)
        elif feature_name == 'lexical':
            param_space['features__lexical__lexical__window_size'] = np.linspace(15, 25, num=2, dtype=np.int)
    clf = GridSearchCV(model, param_space, cv=k_fold, scoring=scorer, n_jobs=-1, refit=False)
    clf.fit(X, y)
    return clf


def build_pipeline(vocab, feature_names, stopwords, cue_word_vocab,
                   cue_bigram_vocab, sent_vocab, lda_matrix):
    """
    Parameters
    ----------
    vocab : {str: int}
    feature_names : [str]
    stopwords : {str}
    cue_words : [str]
    cue_bigrams : [str]
    sent_vocab : {str: int}
    lda_matrix : np.ndarray
        of shape (sent_count, topic_count)

    Returns
    -------
    pipeline : Pipeline
    steps : (str, Estimator)
    """
    features = []
    for feature_name in feature_names:
        if feature_name == 'lexical':
            features.append(('lexical', Pipeline([
                ('bow', BagOfWords(vocab=vocab, stopwords=stopwords)),
                ('tfidf', EstimatorWrapper(estimator=TfidfTransformer(norm='l2', use_idf=True))),
                ('lexical', LexicalFeature(window_size=25))])))
        elif feature_name == 'lda':
            features.append(('lda', LdaLookup(sent_vocab=sent_vocab,
                                              lda_matrix=lda_matrix)))
        elif feature_name == 'depth_score':
            features.append(('depth_score', Pipeline([
                ('bow', BagOfWords(vocab=vocab, stopwords=stopwords)),
                ('tfidf', EstimatorWrapper(
                    estimator=TfidfTransformer(norm='l2', use_idf=True))),
                ('depth_score', DepthScore(block_size=25))])))
        elif feature_name == 'cue_word':
            features.append(('cue_word', CueWord(cue_word_vocab)))
        elif feature_name == 'cue_bigram':
            features.append(('cue_bigram', CueBigram(cue_bigram_vocab)))
        else:
            raise Exception('Invalid feature_name {}'.format(feature_name))

    steps = [('features', CrfFeatureTransformer(
        window_size=4, transformer_list=features, n_jobs=1)),
             ('crf', CRF(algorithm='l2sgd', all_possible_states=True,
                         all_possible_transitions=True))]
    return Pipeline(steps), steps


def to_crf_labels(y):
    output = []
    for y_i_doc in y:
        output.append([unicode(label) for label in y_i_doc])
    return output


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


def main(opt):
    with codecs.open(opt.stopword, encoding='utf-8') as f:
        stopwords = load_set(f)
        logger.info('Load stopwords: %d', len(stopwords))
    with codecs.open(opt.cue_word, encoding='utf-8') as f:
        cue_words = load_list(f)
        cue_word_vocab = dict([(w, i) for i, w in enumerate(cue_words)])
        logger.info('Load cue_words: %d', len(cue_words))
    with codecs.open(opt.cue_bigram, encoding='utf-8') as f:
        cue_bigrams = load_list(f)
        cue_bigram_vocab = dict([(b, i) for i, b in enumerate(cue_bigrams)])
        logger.info('Load cue_bigrams: %d', len(cue_bigrams))
    with codecs.open(opt.vocab, encoding='utf-8') as f:
        vocab = load_vocab(f)
        logger.info('Load vocab: %d', len(vocab))
    with codecs.open(opt.lda_vec, encoding='utf-8') as f:
        sent_vocab, lda_matrix = load_sent_vec(f)
        logger.info('Load lda_vec: %s', lda_matrix.shape)

    ids_train, docs_train, y_train = load_all_data(opt.train_jsons)
    labels_train = to_crf_labels(y_train)

    model, steps = build_pipeline(
            vocab, opt.feature_names, stopwords, cue_word_vocab,
            cue_bigram_vocab, sent_vocab, lda_matrix)
    if opt.search:
        search = param_search(model, opt.feature_names,
                              docs_train, labels_train)
        for params, score, _ in search.grid_scores_:
            logger.info('%r: %f', params, score)
        logger.info('best_params: %r, cv_score: %f',
                    search.best_params_, search.best_score_)
        model.set_params(**search.best_params_)
    else:
        if opt.param_yaml is None:
            raise Exception('No param_yaml given')
        with codecs.open(opt.param_yaml, encoding='utf-8') as f:
            params = yaml.load(f)
            model.set_params(**params)
            logger.info('Load and set param: %s', params)

    model.fit(docs_train, labels_train)

    print("Top likely transitions:")
    print_transitions(Counter(model.named_steps['crf'].transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(model.named_steps['crf'].transition_features_).most_common()[-20:])

    print("Top positive:")
    print_state_features(Counter(model.named_steps['crf'].state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(model.named_steps['crf'].state_features_).most_common()[-30:])

    labels_train_pred = model.predict(docs_train)

    window_diffs, average_wd = average_window_diff(labels_train, labels_train_pred, boundary_label='1')
    logger.info('on train data, wds: %s, average: %f',
                window_diffs, average_wd)

    if opt.train_segment:
        with codecs.open(opt.train_segment, mode='w', encoding='utf-8') as f:
            pred_to_segment_jsonl(ids_train, labels_train_pred, f, boundary_label='1')

    ids_test, docs_test, y_test = load_all_data(opt.test_jsons)
    labels_test = to_crf_labels(y_test)
    labels_test_pred = model.predict(docs_test)
    pks, average_pk_ = average_pk(labels_test, labels_test_pred, boundary_label='1')
    logger.info('on test data, pks: %s', pks)
    window_diffs, average_wd = average_window_diff(labels_test, labels_test_pred, boundary_label='1')
    logger.info('on test data, wds: %s', window_diffs)
    logger.info('on test data, average_pk: %f, average_wd: %f',
                average_pk_, average_wd)
    if opt.test_segment:
        with codecs.open(opt.test_segment, mode='w', encoding='utf-8') as f:
            pred_to_segment_jsonl(ids_test, labels_test_pred, f, boundary_label='1')
    sys.exit(0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--feature_names', nargs='+')
    arg_parser.add_argument('--search', action='store_true')
    arg_parser.add_argument('--param_yaml', default=None)

    arg_parser.add_argument('--vocab')
    arg_parser.add_argument('--stopword')
    arg_parser.add_argument('--cue_word')
    arg_parser.add_argument('--cue_bigram')
    arg_parser.add_argument('--lda_vec')

    arg_parser.add_argument('--train_jsons', nargs='+')
    arg_parser.add_argument('--test_jsons', nargs='+')
    arg_parser.add_argument('--train_segment')
    arg_parser.add_argument('--test_segment')
    opt = arg_parser.parse_args()

    opt_print = dict(vars(opt))
    opt_print['train_jsons'] = len(opt_print['train_jsons'])
    opt_print['test_jsons'] = len(opt_print['test_jsons'])
    logger.info('opts: %r', opt_print)

    main(opt)
