# coding: utf-8

import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
import argparse
import sys
import logging
import yaml

from evaluation import average_window_diff, window_diff_scorer, \
        average_pk
from resource import load_all_data, load_set, load_list, load_vocab, \
        load_sent_vec
from feature import SentToGap, CueWord, CueBigram, DepthScore, \
        BagOfWords, EstimatorWrapper, FeatureUnion, LdaLookup, \
        LexicalFeature
from json_core import pred_to_segment_jsonl


logger = logging.getLogger(__name__)


def build_id2word(vocab):
    return dict([(id_, w) for w, id_ in vocab.iteritems()])


def grid_search(model, feature_names, final_model_name, X, y):
    scorer = window_diff_scorer(punish_zero_seg=True)
    k_fold = KFold(len(y), n_folds=5, shuffle=True, random_state=721)
    grid = {}
    for feature_name in feature_names:
        if feature_name == 'lexical':
            grid['features__lexical__lexical__window_size'] = np.linspace(5, 25, num=3, dtype=np.int)
        elif feature_name == 'depth_score':
            grid['features__depth_score__depth_score__block_size'] = np.linspace(5, 25, num=3, dtype=np.int)
    if final_model_name == 'rbf_svm':
        grid['final__C'] = [1e-1, 1, 1e1]
        grid['final__gamma'] = np.logspace(-1, 1, num=3, base=2.0)
    elif final_model_name == 'linear_svm':
        grid['final__C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
    clf = GridSearchCV(model, grid, cv=k_fold, scoring=scorer, n_jobs=-1, refit=False)
    clf.fit(X, y)
    return clf


def build_pipeline(vocab, feature_names, final_model_name, stopwords,
                   cue_word_vocab, cue_bigram_vocab, sent_vocab, lda_matrix):
    """
    Parameters
    ----------
    vocab : {str: int}
    feature_names : [str]
    final_model_name : str
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
            features.append(('lda', Pipeline([
                ('lda', LdaLookup(sent_vocab=sent_vocab, lda_matrix=lda_matrix)),
                ('sent2gap', SentToGap())])))
        elif feature_name == 'depth_score':
            features.append(('depth_score', Pipeline([
                ('bow', BagOfWords(vocab=vocab, stopwords=stopwords)),
                ('tfidf', EstimatorWrapper(estimator=TfidfTransformer(norm='l2', use_idf=True))),
                ('depth_score', DepthScore(block_size=15))])))
        elif feature_name == 'cue_word':
            features.append(('cue_word', Pipeline([
                ('cue_word', CueWord(cue_word_vocab)),
                ('sent2gap', SentToGap())])))
        elif feature_name == 'cue_bigram':
            features.append(('cue_bigram', Pipeline([
                ('cue_bigram', CueBigram(cue_bigram_vocab)),
                ('sent2gap', SentToGap())])))
        else:
            raise Exception('Invalid feature_name {}'.format(feature_name))

    steps = [('features', FeatureUnion(features, n_jobs=1))]
    if final_model_name == 'rbf_svm':
        steps.append(('final', EstimatorWrapper(estimator=SVC(kernel='rbf', class_weight='balanced'))))
    elif final_model_name == 'linear_svm':
        steps.append(('final', EstimatorWrapper(estimator=LinearSVC(C=100, class_weight='balanced', random_state=721))))
    else:
        raise Exception('invalid final_model_name {}'.format(final_model_name))
    return Pipeline(steps), steps


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

    model, steps = build_pipeline(
            vocab, opt.feature_names, opt.final_model_name, stopwords, cue_word_vocab,
            cue_bigram_vocab, sent_vocab, lda_matrix)
    if opt.grid_search:
        search = grid_search(model, opt.feature_names, opt.final_model_name, docs_train, y_train)
        for params, score, _ in search.grid_scores_:
            logger.info('%r: %f', params, score)
        logger.info('best_params: %r, cv_score: %f', search.best_params_, search.best_score_)
        model.set_params(**search.best_params_)
    else:
        if opt.param_yaml is None:
            raise Exception('No param_yaml given')
        with codecs.open(opt.param_yaml, encoding='utf-8') as f:
            params = yaml.load(f)
            model.set_params(**params)
            logger.info('Load and set param: %s', params)

    model.fit(docs_train, y_train)
    y_train_pred = model.predict(docs_train)
    window_diffs, average_wd = average_window_diff(y_train, y_train_pred)
    logger.info('on train data, wds: %s, average: %f', window_diffs, average_wd)

    if opt.train_segment:
        with codecs.open(opt.train_segment, mode='w', encoding='utf-8') as f:
            pred_to_segment_jsonl(ids_train, y_train_pred, f)

    ids_test, docs_test, y_test = load_all_data(opt.test_jsons)
    y_pred = model.predict(docs_test)
    pks, average_pk_ = average_pk(y_test, y_pred)
    logger.info('on test data, pks: %s', pks)
    window_diffs, average_wd = average_window_diff(y_test, y_pred)
    logger.info('on test data, wds: %s', window_diffs)
    logger.info('on test data, average_pk: %f, average_wd: %f', average_pk_, average_wd)
    if opt.test_segment:
        with codecs.open(opt.test_segment, mode='w', encoding='utf-8') as f:
            pred_to_segment_jsonl(ids_test, y_pred, f)
    sys.exit(0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--feature_names', nargs='+')
    arg_parser.add_argument('--final_model_name')
    arg_parser.add_argument('--grid_search', action='store_true')
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
