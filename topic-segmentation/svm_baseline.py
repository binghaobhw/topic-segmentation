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
from time import time

from evaluation import average_window_diff, window_diff_scorer
from json_core import pred_to_segment_jsonl
from resource import load_all_data, load_set, load_vocab
from feature import BagOfWords, LexicalFeature, EstimatorWrapper


logger = logging.getLogger(__name__)

np.random.seed(721)


def average_non_zero_feature_percentage(X):
    """
    Parameter
    ---------
    X : [matrix]

    Returns
    -------
    percentage : float
    """
    if not X:
        raise Exception('empty X')
    averages = map(lambda x: np.average(x.getnnz(axis=1)), X)
    return np.average(averages) / X[0].shape[1]


def grid_search(model, grid_name, X, y):
    scorer = window_diff_scorer(punish_zero_seg=True)
    k_fold = KFold(len(y), n_folds=5, shuffle=True, random_state=721)
    grid = {}
    if grid_name == 'rbf_svm':
        grid['final__C'] = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        grid['final__gamma'] = np.logspace(-6, 6, num=13, base=2.0)
    elif grid_name == 'linear_svm':
        grid['final__C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
    elif grid_name == 'window':
        grid['lexical__window_size'] = np.linspace(5, 50, num=10, dtype=np.int)
    else:
        raise Exception('invalid grid_name {}'.format(grid_name))
    clf = GridSearchCV(model, grid, cv=k_fold, scoring=scorer, n_jobs=-1, refit=False)
    clf.fit(X, y)
    return clf


def build_pipeline(vocab, final_model_name, stopwords):
    """
    Parameters
    ----------
    vocab : {str: int}
    final_model_name : str
    stopwords : {str}

    Returns
    -------
    pipeline : Pipeline
    steps : (str, Estimator)
    """
    steps = [
            ('bow', BagOfWords(vocab=vocab, stopwords=stopwords)),
            ('tfidf', EstimatorWrapper(estimator=TfidfTransformer(norm='l2', use_idf=True))),
            ('lexical', LexicalFeature(window_size=25))]
    if final_model_name == 'rbf_svm':
        steps.append(('final', EstimatorWrapper(estimator=SVC(kernel='rbf', class_weight='balanced'))))
    elif final_model_name == 'linear_svm':
        steps.append(('final', EstimatorWrapper(estimator=LinearSVC(C=1, class_weight='balanced', random_state=721))))
    else:
        raise Exception('invalid final_model_name {}'.format(final_model_name))
    return Pipeline(steps), steps


def main(opt):
    with codecs.open(opt.stopword, encoding='utf-8') as f:
        stopwords = load_set(f)
    with codecs.open(opt.vocab, encoding='utf-8') as f:
        vocab = load_vocab(f)

    ids_train, docs_train, y_train = load_all_data(opt.train_jsons)

    model, steps = build_pipeline(vocab, opt.final_model_name, stopwords)
    if opt.grid_name:
        # transformers = Pipeline(steps[:-1])
        # X_train_trans = transformers.fit_transform(X_train, y_train)
        # logger.info('average_non_zero_feature_percentage for train: %f',
        #             average_non_zero_feature_percentage(X_train_trans))
        search = grid_search(model, opt.grid_name, docs_train, y_train)
        for params, score, _ in search.grid_scores_:
            logger.info('%r: %f', params, score)
        logger.info('best_params: %r, cv_score: %f', search.best_params_, search.best_score_)
        model.set_params(**search.best_params_)

    model.fit(docs_train, y_train)
    y_train_pred = model.predict(docs_train)
    window_diffs, average_wd = average_window_diff(y_train, y_train_pred)
    logger.info('on train data, wds: %s, average: %f', window_diffs, average_wd)

    if opt.train_segment:
        with codecs.open(opt.train_segment, mode='w', encoding='utf-8') as f:
            pred_to_segment_jsonl(ids_train, y_train_pred, f)

    ids_test, docs_test, y_test = load_all_data(opt.test_jsons)
    y_pred = model.predict(docs_test)
    window_diffs, average_wd = average_window_diff(y_test, y_pred)
    logger.info('on test data, wds: %s, average: %f', window_diffs, average_wd)
    if opt.test_segment:
        with codecs.open(opt.test_segment, mode='w', encoding='utf-8') as f:
            pred_to_segment_jsonl(ids_test, y_pred, f)
    sys.exit(0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--final_model_name')
    arg_parser.add_argument('--grid_name', default=None)

    arg_parser.add_argument('--vocab')
    arg_parser.add_argument('--stopword')

    arg_parser.add_argument('--train_jsons', nargs='+')
    arg_parser.add_argument('--test_jsons', nargs='+')
    arg_parser.add_argument('--train_segment')
    arg_parser.add_argument('--test_segment')
    opt = arg_parser.parse_args()
    logger.info('opts: %r', opt)
    main(opt)
