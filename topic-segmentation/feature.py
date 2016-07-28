# coding: utf-8

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cosine
from gensim.models.ldamulticore import LdaMulticore
from gensim.matutils import Sparse2Corpus
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import FeatureUnion as SkFeatureUnion
from sklearn.externals.joblib import Parallel, delayed
import logging

from resource import EMPTY_TOKEN

np.random.seed(721)

logger = logging.getLogger(__name__)


class BagOfWords(BaseEstimator, TransformerMixin):
    def __init__(self, vocab=None, stopwords=None):
        self.vocab = vocab
        self.stopwords = stopwords

    def fit(self, docs, y=None):
        """
        Parameter
        ---------
        docs : [[[str]]]

        Returns
        -------
        self
        """
        return self

    def fit_transform(self, docs, y=None):
        """
        Parameter
        ---------
        docs : [[[str]]]

        Returns
        -------
        matrices : [sp.csr_matrix]
            matrices[i] is sent-term count matrix for i-th doc
        """
        self.fit(docs, y)
        return self.transform(docs)

    def transform(self, docs):
        """
        Parameter
        ---------
        docs : [[[str]]]

        Returns
        -------
        matrices : [sp.csr_matrix]
            matrices[i] is line-term count matrix for i-th doc
        """
        if not self.vocab:
            raise Exception('emtpy vocab')
        output = []
        for doc in docs:
            col_indices = []  # Contains col index in matrix of each value of data
            value_indices = [0]  # len = row_num + 1
            for sent in doc:
                filtered = filter(lambda w: w in self.vocab, sent)
                if len(filtered) == 0:
                    filtered.append(EMPTY_TOKEN)
                for word in filtered:
                    col_indices.append(self.vocab[word])
                value_indices.append(len(col_indices))
            values = np.ones(len(col_indices))
            line_term_matrix = sp.csr_matrix((values, col_indices, value_indices),
                                             shape=(len(value_indices) - 1, len(self.vocab)),
                                             dtype=np.int)
            line_term_matrix.sum_duplicates()
            output.append(line_term_matrix)
        return output


class LexicalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=1):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        """
        Parameter
        ---------
        X : [matrix]
            each matrix is of shape (sent_count, vocab_size)

        Returns
        -------
        out : [matrix]
            each matrix is of shape (gap_count, vocab_size)
        """
        output = []
        for doc in X:
            col_indices = []  # Contains col index in matrix of each value of data
            value_indices = [0]  # len = row_num + 1
            values = []
            n_samples, n_features = doc.shape
            gap_count = n_samples - 1
            for i in xrange(gap_count):
                left_most = i - self.window_size
                left_border = left_most if left_most > 0 else 0
                left_window = doc[left_border:i+1]

                right_most = i + self.window_size
                right_border = right_most if right_most < gap_count else gap_count
                right_window = doc[i+1:right_border+1]

                left_sum = left_window.sum(axis=0)
                right_sum = right_window.sum(axis=0)

                product = np.multiply(left_sum, right_sum)
                nonzero_indices = np.nonzero(product)  # (array, array)
                nonzero_col_indices = nonzero_indices[1].tolist()  # []
                if nonzero_col_indices:
                    # col index
                    col_indices.extend(nonzero_col_indices)
                    values.extend(product[nonzero_indices].tolist()[0])
                value_indices.append(len(col_indices))
            output.append(sp.csr_matrix((values, col_indices, value_indices),
                                        shape=(gap_count, n_features),
                                        dtype=doc.dtype))
        return output


class EstimatorWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def set_params(self, **params):
        params_ = {}
        for k, v in params.iteritems():
            if k == 'estimator':
                continue
            params_[k] = v
        self.estimator.set_params(**params_)

    def get_params(self, deep=True):
        if not deep:
            return super(EstimatorWrapper, self).get_params(deep=False)
        params = super(EstimatorWrapper, self).get_params(deep=False)
        params.update(self.estimator.get_params(deep=True))
        return params

    def _flattern_X(self, X):
        if any(sp.issparse(m) for m in X):
            X_flat = sp.vstack(X)
        else:
            X_flat = np.vstack(X)
        return X_flat

    def transform(self, X):
        """
        Parameter
        ---------
        X : [matrix]

        Returns
        -------
        output : [matrix]
        """
        X_flat = self._flattern_X(X)
        flat_out = self.estimator.transform(X_flat)

        output = []
        base = 0
        for doc in X:
            len_ = doc.shape[0]
            output.append(flat_out[base:base+len_])
            base += len_
        return output

    def fit_transform(self, X, y=None):
        X_flat = self._flattern_X(X)
        y_flat = np.concatenate(y) if y else None
        matrix = self.estimator.fit_transform(X_flat, y_flat)

        output = []
        base = 0
        for i in xrange(len(X)):
            doc = X[i]
            len_ = doc.shape[0]
            output.append(matrix[base:base+len_])
            base += len_
        return output

    def fit(self, X, y=None):
        """
        Parameter
        ---------
        X : [matrix or list]
        y : [np.ndarray] or None

        Returns
        -------
        self
        """
        X_flat = self._flattern_X(X)
        if y:
            y_flat = np.concatenate(y)
            self.estimator.fit(X_flat, y_flat)
        else:
            self.estimator.fit(X_flat)

        return self

    def predict(self, X):
        """
        Parameter
        ---------
        X : [matrix]

        Returns
        -------
        pred : [np.ndarray]
        """
        pred = []
        for doc in X:
            pred.append(self.estimator.predict(doc))
        return pred


class SentToGap(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        """
        Parameter
        ---------
        X : [matrix]
            each matrix is of shape (sent_count, feature_count)

        Returns
        -------
        out : [matrix]
            each matrix is of shape (gap_count, feature_count)
        """
        return [m[1:] for m in X]


class CueBigram(BaseEstimator, TransformerMixin):
    def __init__(self, cue_bigram_vocab=None):
        """
        Parameter
        ---------
        cue_bigrams : [str]

        """
        self.cue_bigram_vocab = cue_bigram_vocab

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, docs):
        """
        Parameter
        ---------
        docs : [[[str]]]

        Returns
        -------
        out : [np.ndarray]
            out[i] shape (sent_count, cue_bigram_count)
            out[i][j][k] = 1 if k-th cue bigram exists for j-th sent in i-th doc,
            else = 0
        """
        out = []
        cue_bigram_count = len(self.cue_bigram_vocab)
        for doc in docs:
            features = np.zeros((len(doc), cue_bigram_count), dtype=np.float32)
            for i, sent in enumerate(doc):
                sent_len = len(sent)
                if sent_len < 2:
                    continue
                for i in xrange(sent_len - 1):
                    bigram = ' '.join(sent[i:i + 2])
                    if bigram in self.cue_bigram_vocab:
                        features[i][self.cue_bigram_vocab[bigram]] = 1
            out.append(features)
        return out


class CueWord(BaseEstimator, TransformerMixin):
    def __init__(self, cue_word_vocab=None):
        """
        Parameter
        ---------
        cue_words : [str]

        """
        self.cue_word_vocab = cue_word_vocab

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, docs):
        """
        Parameter
        ---------
        docs : [[[str]]]

        Returns
        -------
        out : [np.ndarray]
            out[i] shape (doc_len, cue_word_count)
            out[i][j][k] = 1 if k-th cue word exists for j-th sent in i-th doc,
            else = 0
        """
        out = []
        cue_word_count = len(self.cue_word_vocab)
        for doc in docs:
            features = np.zeros((len(doc), cue_word_count), dtype=np.float32)
            for i, sent in enumerate(doc):
                for word in sent:
                    if word in self.cue_word_vocab:
                        features[i][self.cue_word_vocab[word]] = 1
            out.append(features)
        return out


class Lda(BaseEstimator, TransformerMixin):
    def __init__(self, id2word=None, num_topics=25, passes=1):
        self.lda = None
        self.id2word = id2word
        self.num_topics = num_topics
        self.passes = passes

    def fit(self, X, y=None):
        """
        Parameter
        ---------
        X : [sp.csr_matrix]

        Returns
        -------
        self
        """
        if self.lda is None:
            self.lda = LdaMulticore(
                    id2word=self.id2word, num_topics=self.num_topics, passes=self.passes)
        X_flat = sp.vstack(X)
        self.lda.update(Sparse2Corpus(X_flat, documents_columns=False))
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        Parameter
        ---------
        X : [sp.csr_matrix]

        Returns
        -------
        topic_vectors : [np.ndarray]
            each matrix is of shape (sent_count, topic_count)
        """
        topic_vectors = []
        for doc in X:
            sents_bow = Sparse2Corpus(doc, documents_columns=False)
            gamma, _ = self.lda.inference(sents_bow)
            # divide row by row sum
            topic_dist = (gamma.T / np.sum(gamma, axis=1)).T
            topic_vectors.append(topic_dist)
        return topic_vectors


class LdaLookup(BaseEstimator, TransformerMixin):
    def __init__(self, sent_vocab=None, lda_matrix=None):
        self.sent_vocab = sent_vocab
        self.lda_matrix = lda_matrix

    def fit(self, X, y=None):
        """
        Parameter
        ---------
        X : [[[str]]]

        Returns
        -------
        self
        """
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        Parameter
        ---------
        X : [[[str]]]

        Returns
        -------
        topic_vectors : [np.ndarray]
            each matrix is of shape (sent_count, topic_count)
        """
        topic_vectors = []
        for doc in X:
            sent_ids = []
            for sent in doc:
                sent_text = ' '.join(sent)
                sent_id = self.sent_vocab[sent_text]
                sent_ids.append(sent_id)
            topic_vectors.append(self.lda_matrix[sent_ids])
        return topic_vectors


class DepthScore(BaseEstimator, TransformerMixin):
    def __init__(self, block_size=5):
        self.block_size = block_size

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        Parameter
        ---------
        X : [matrix]
            each term weight matrix is of shape (sent_count, vocab_count)

        Returns
        -------
        out : [np.ndarray]
            each array is of shape (gap_count, 1)
        """
        out = []
        for matrix in X:
            gap_scores = self._smooth(self._gap_scores(matrix))
            depth_scores = self._depth_scores(gap_scores)
            out.append(depth_scores.reshape((depth_scores.shape[0], 1)))
        return out

    def _gap_scores(self, matrix):
        sent_count = matrix.shape[0]
        gap_count = sent_count - 1

        gap_scores = np.empty((gap_count,), dtype=np.float32)
        for i_gap in xrange(gap_count):
            if i_gap + 1 < self.block_size:
                block_size = i_gap + 1
            elif gap_count - i_gap < self.block_size:
                block_size = gap_count - i_gap
            else:
                block_size = self.block_size
            left = matrix[i_gap-block_size+1:i_gap+1].sum(axis=0)
            right = matrix[i_gap+1:i_gap+block_size+1].sum(axis=0)
            gap_scores[i_gap] = cosine(left, right)
        return gap_scores

    def _smooth(self, gap_scores):
        return gap_scores

    def _depth_scores(self, gap_scores):
        depth_scores = np.empty((len(gap_scores),), dtype=np.float32)
        for i, current_score in enumerate(gap_scores):
            lpeak = current_score
            for score in gap_scores[i::-1]:
                if score >= lpeak:
                    lpeak = score
                else:
                    break
            rpeak = current_score
            for score in gap_scores[i:]:
                if score >= rpeak:
                    rpeak = score
                else:
                    break
            depth_scores[i] = (lpeak-current_score) + (rpeak-current_score)
        return depth_scores


def _transform_one(transformer, X):
    return transformer.transform(X)


def _fit_transform_one(transformer, X, y, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        X_transformed = transformer.fit_transform(X, y, **fit_params)
        return X_transformed, transformer
    else:
        X_transformed = transformer.fit(X, y, **fit_params).transform(X)
        return X_transformed, transformer


class FeatureUnion(SkFeatureUnion):
    def __init__(self, transformer_list=None, n_jobs=1):
        super(FeatureUnion, self).__init__(
                transformer_list, n_jobs=n_jobs)

    def fit_transform(self, X, y=None, **fit_params):
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, **fit_params)
            for name, trans in self.transformer_list)

        transformed_list, transformers = zip(*result)
        super(FeatureUnion, self)._update_transformer_list(transformers)
        return self._concat(transformed_list, len(X))

    def transform(self, X):
        """
        Parameter
        ---------
        X : [[[str]]]
        """
        # each one is [matrix]
        transformed_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X)
            for name, trans in self.transformer_list)
        return self._concat(transformed_list, len(X))

    def _concat(self, transformered_list, doc_count):
        out = []
        for i_doc in xrange(doc_count):
            all_matrices_i_doc = [matrices[i_doc]
                                  for matrices in transformered_list]
            if any(sp.issparse(m) for m in all_matrices_i_doc):
                Xs = sp.hstack(all_matrices_i_doc).tocsr()
            else:
                Xs = np.hstack(all_matrices_i_doc)
            out.append(Xs)
        return out


class CrfFeatureTransformer(SkFeatureUnion):
    def __init__(self, window_size=2, transformer_list=None, n_jobs=1):
        """
        Parameter
        ---------
        window_size : int
        transformer_list : [(str, Estimator)]
        n_jobs : int
        """
        self.window_size = window_size
        super(CrfFeatureTransformer, self).__init__(
                transformer_list, n_jobs=n_jobs)

    def fit_transform(self, X, y=None, **fit_params):
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, **fit_params)
            for name, trans in self.transformer_list)

        transformed_list, transformers = zip(*result)
        super(CrfFeatureTransformer, self)._update_transformer_list(transformers)

        doc_count = len(X)
        gap_counts = [len(X_i) - 1 for X_i in X]
        return self._to_crf_format(transformed_list, doc_count, gap_counts)

    def transform(self, X):
        transformed_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X)
            for name, trans in self.transformer_list)

        doc_count = len(X)
        gap_counts = [len(X_i) - 1 for X_i in X]
        return self._to_crf_format(transformed_list, doc_count, gap_counts)

    def _features2key_value(self, features, prefix):
        """
        Parameter
        ---------
        features : np.ndarray or sparse_matrix
            shape (feature_count,) for np.ndarray
            shape (1, feature_count) for np.ndarray
        """
        output = {}
        if isinstance(features, sp.csr_matrix):
            feature_count = features.shape[1]
            for i in xrange(feature_count):
                v = features[0, i]
                output['{}[{}]'.format(prefix, i)] = v  # as weight
        else:
            for i, v in enumerate(features):
                output['{}[{}]'.format(prefix, i)] = v  # as weight
        return output

    def _gap2features(self, i_gap, matrices, gap_count, feature_names):
        left_i_gap = i_gap - self.window_size
        if left_i_gap < 0:
            left_i_gap = 0
        right_i_gap = i_gap + self.window_size
        if right_i_gap >= gap_count:
            right_i_gap = gap_count - 1

        features = {}
        # for each gap in a window
        positions = range(left_i_gap, right_i_gap + 1)
        # 0 is name of the current gap
        position_names = map(unicode, xrange(
            0 - self.window_size, self.window_size + 1))
        for i, name in zip(positions, position_names):
            for matrix, feature_name in zip(matrices, feature_names):
                prefix = 'g[{}]_{}'.format(name, feature_name)
                row_count = matrix.shape[0]
                if row_count == gap_count:
                    i_row = i
                elif row_count == gap_count + 1:
                    i_row = i + 1
                else:
                    raise Exception('matrix shape[0] {} not valid, gap_count {}'.format(
                            row_count, gap_count))
                features.update(self._features2key_value(matrix[i_row], prefix))
        return features

    def _doc2features(self, matrices, gap_count, feature_names):
        return [self._gap2features(i_gap, matrices, gap_count, feature_names)
                for i_gap in xrange(gap_count)]

    def _to_crf_format(self, transformered_list, doc_count, gap_counts):
        out = []
        feature_names = [name for name, _ in self.transformer_list]
        for i_doc in xrange(doc_count):
            all_matrices_i_doc = [matrices[i_doc]
                                  for matrices in transformered_list]
            gap_count = gap_counts[i_doc]
            doc_features = self._doc2features(all_matrices_i_doc, gap_count, feature_names)
            out.append(doc_features)
        return out
