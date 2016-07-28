#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import argparse
import json
import codecs
import time
import cPickle as pickle
import sys
import yaml
import logging.config

from json_core import Segment, append_to_jsonl
from resource import load_word_vec, load_vocab_from_list

logger = logging.getLogger()

np.random.seed(721)

EMPTY_TOKEN = '<empty>'
NEW_START_CLASS_INDEX = 0


def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    with open(filename, 'w') as f:
        pickle.dump(data, f)


def load_data(json_file, vocab, speaker_vocab):
    """
    Parameters
    ----------
    json_file : readable
    vocab : {str: int}
    speaker_vocab : {str: int}

    Returns
    -------
    id_ : str
        doc name
    X : np.ndarray
        shape (sent_count, max_sent_len, word_vec_size)
    sent_mask : np.ndarray
        shape (sent_count, max_sent_len)
    speaker_ids : np.ndarray
        shape (sent_count,)
    y : np.ndarray
        shape (sent_count,), y[i] = 0 means i-th sent is the new_start sent.

        classes | new_start | not_new_start
        ------- | --------- | -------------
        index   | 0         | 1
    """
    def load_start_sent_indices(topics):
        return map(lambda t: t['end'] + 1, topics[:-1])

    doc = json.load(json_file)
    id_ = doc['id']
    start_sent_indices = load_start_sent_indices(doc['topics'])
    word_ids = []
    sents = doc['sents']
    sent_count = len(sents)
    max_sent_len = 0
    speaker_ids = np.empty((sent_count,), dtype='int32')
    for i, sent in enumerate(sents):
        speaker_ids[i] = speaker_vocab[sent['speaker']]
        words = filter(lambda w: w in vocab, sent['text'].lower().split(' '))
        if len(words) == 0:
            words.append(EMPTY_TOKEN)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        word_ids.append(map(lambda w: vocab[w], words))

    X = np.zeros((sent_count, max_sent_len), dtype='int32')
    sent_mask = np.zeros((sent_count, max_sent_len), dtype=theano.config.floatX)
    for i_sent, sent_word_ids in enumerate(word_ids):
        for i_word, word_id in enumerate(sent_word_ids):
            X[i_sent, i_word] = word_id
            sent_mask[i_sent, i_word] = 1

    y = np.ones((sent_count,), dtype='int32')
    for i in start_sent_indices:
        y[i] = 0

    return id_, X, sent_mask, speaker_ids, y


def iter_data(json_paths, vocab, speaker_vocab):
    """
    Parameters
    ----------
    json_paths : [str]
    vocab : {str: int}
    speaker_vocab : {str: int}

    Returns
    -------
    id_ : str
        doc name
    X : np.ndarray
        shape (sent_count, max_sent_len, word_vec_size)
    sent_mask : np.ndarray
        shape (sent_count, max_sent_len)
    speaker_ids : np.ndarray
        shape (sent_count,)
    y : np.ndarray
        shape (sent_count,)
    """
    for json_path in json_paths:
        with codecs.open(json_path, encoding='utf-8') as f:
            yield load_data(f, vocab, speaker_vocab)


def to_segment(id_, y, jsonl_file):
    """
    Parameters
    ----------
    id_ : str
        doc name
    y : np.ndarray
        shape (sent_count, 2)
    jsonl_file : writable
    """
    max_indices = np.argmax(y, axis=1)
    new_start_sent_indices = np.where(max_indices == NEW_START_CLASS_INDEX)[0].tolist()
    boundary_indices = map(lambda x: x - 1, new_start_sent_indices)
    segment = Segment(id_, boundary_indices, y.shape[0] - 1)
    append_to_jsonl(segment, jsonl_file)


def main(opt):
    with codecs.open(opt.word_vec_path, encoding='utf-8') as f:
        vocab, word_matrix = load_word_vec(f)
    word_vec_size = word_matrix.shape[1]

    with codecs.open(opt.speaker_vocab_path, encoding='utf-8') as f:
        speaker_vocab = load_vocab_from_list(f)
    # (sent_count, max_sent_len)
    X_var = T.imatrix('X')
    input_layer = lasagne.layers.InputLayer((None, None), input_var=X_var)
    sent_input_mask_var = T.fmatrix('sent_input_mask')
    sent_input_mask_layer = lasagne.layers.InputLayer(
            (None, None), input_var=sent_input_mask_var)

    word_vec_layer = lasagne.layers.EmbeddingLayer(
            input_layer, input_size=len(vocab),
            output_size=word_vec_size, W=word_matrix)
    if not opt.train_word_vec:
        word_vec_layer.params[word_vec_layer.W].remove('trainable')

    dropout_input_layer = lasagne.layers.DropoutLayer(
            word_vec_layer, p=opt.dropout)

    # output shape (sent_count, sent_vec_size)
    sent_layer = lasagne.layers.LSTMLayer(
            dropout_input_layer, opt.sent_vec_size,
            grad_clipping=opt.grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh,
            backwards=opt.sent_backwards,
            precompute_input=True,
            mask_input=sent_input_mask_layer, only_return_final=True)
    if opt.speaker_vec_size > 0:
        speaker_ids_var = T.ivector('speaker_ids')  # speaker id vector
        speaker_input_layer = lasagne.layers.InputLayer(
                (None,), input_var=speaker_ids_var)
        # output shape (sent_count, speaker_vec_size)
        speaker_layer = lasagne.layers.EmbeddingLayer(
                speaker_input_layer, input_size=len(speaker_vocab),
                output_size=opt.speaker_vec_size)
        # output shape (sent_count, sent_vec_size + speaker_vec_size)
        sent_layer = lasagne.layers.ConcatLayer(
                [sent_layer, speaker_layer], axis=1)

    # output shape (1, sent_count, sent_vec_size + speaker_vec_size)
    sent_reshape_layer = lasagne.layers.ReshapeLayer(
            sent_layer, (-1, [0], [1]))

    # output shape (1, sent_count, hidden_vec_size)
    doc_layer = lasagne.layers.LSTMLayer(
            sent_reshape_layer, opt.hidden_vec_size,
            grad_clipping=opt.grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh,
            precompute_input=True)
    # output shape (sent_count, hidden_vec_size)
    doc_reshape_layer = lasagne.layers.ReshapeLayer(doc_layer, (-1, [2]))

    # if opt.speaker_vec_size > 0:
    #     speaker_ids_var = T.ivector('speaker_ids')  # speaker id vector
    #     speaker_input_layer = lasagne.layers.InputLayer(
    #             (None,), input_var=speaker_ids_var)
    #     # output shape (sent_count, speaker_vec_size)
    #     speaker_layer = lasagne.layers.EmbeddingLayer(
    #             speaker_input_layer, input_size=len(speaker_vocab),
    #             output_size=opt.speaker_vec_size)
    #     concat_layer = lasagne.layers.ConcatLayer(
    #             [doc_reshape_layer, speaker_layer], axis=1)
    #     # output shape (sent_count, 2)
    #     output_layer = lasagne.layers.DenseLayer(
    #             concat_layer, num_units=2,
    #             nonlinearity=lasagne.nonlinearities.softmax)
    # else:
    #     # output shape (sent_count, 2)
    #     output_layer = lasagne.layers.DenseLayer(
    #             doc_reshape_layer, num_units=2,
    #             nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = lasagne.layers.DenseLayer(
            doc_reshape_layer, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    net_output = lasagne.layers.get_output(output_layer)

    # (sent_count,)
    reference = T.ivector('reference')
    loss = lasagne.objectives.categorical_crossentropy(net_output, reference).sum()

    all_params = lasagne.layers.get_all_params(output_layer)

    updates = lasagne.updates.rmsprop(
            loss, all_params, learning_rate=opt.learning_rate)

    if opt.speaker_vec_size > 0:
        train_vars = [X_var, sent_input_mask_var, speaker_ids_var, reference]
        test_vars = [X_var, sent_input_mask_var, speaker_ids_var]
    else:
        train_vars = [X_var, sent_input_mask_var, reference]
        test_vars = [X_var, sent_input_mask_var]

    logger.info('Compiling train function ...')
    start_time = time.time()
    train = theano.function(train_vars, loss, updates=updates)
    valid = theano.function(train_vars, loss)
    logger.info('Compiling train function took %.3fs', time.time() - start_time)

    if opt.load_model:
        logger.info('Loading model ...')
        start_time = time.time()
        read_model_data(output_layer, opt.model_path)
        logger.info('Loading model took %.3fs', time.time() - start_time)

    best_loss = np.inf
    best_epoch = -1
    epoch = 0
    loop = True
    while (epoch < opt.epoch) and loop:
        start_time = time.time()

        loss_train = 0
        for id_, X_train, sent_mask_train, speaker_ids, y_train in iter_data(opt.train_jsons, vocab, speaker_vocab):
            if opt.speaker_vec_size > 0:
                loss_train += train(X_train, sent_mask_train, speaker_ids, y_train)
            else:
                loss_train += train(X_train, sent_mask_train, y_train)
        if loss_train > 0 and len(opt.train_jsons) > 0:
            loss_train = loss_train / len(opt.train_jsons)

        loss_valid = 0
        for id_, X_valid, sent_mask_valid, speaker_ids, y_valid in iter_data(opt.valid_jsons, vocab, speaker_vocab):
            if opt.speaker_vec_size > 0:
                loss_valid += valid(X_valid, sent_mask_valid, speaker_ids, y_valid)
            else:
                loss_valid += valid(X_valid, sent_mask_valid, y_valid)
        if loss_valid > 0 and len(opt.valid_jsons) > 0:
            loss_valid = loss_valid / len(opt.valid_jsons)

        logger.info("Epoch %d took %.3fs, train loss = %.6f, valid loss = %.6f",
                    epoch, time.time() - start_time, loss_train, loss_valid)

        improvement = best_loss - loss_valid
        if best_epoch == -1 or improvement > opt.min_improvement:
            best_epoch = epoch
            best_loss = loss_valid
            write_model_data(output_layer, opt.model_path)
            logger.info('Save new best model: epoch %d, loss = %.6f', epoch, loss_valid)

        if epoch >= best_epoch + opt.max_epoch_no_improvement:
            loop = False
            logger.info('Early stop')
        # Important
        epoch += 1

    logger.info('Compiling test function ...')
    start_time = time.time()
    # Disabling dropout layers when test
    test_output = lasagne.layers.get_output(output_layer, deterministic=True)
    test = theano.function(test_vars, test_output)
    logger.info('Compiling test function took %.3fs', time.time() - start_time)

    logger.info('Test ...')
    start_time = time.time()
    with codecs.open(opt.segment_jsonl, mode='w', encoding='utf-8') as jsonl_file:
        for id_, X_test, sent_mask_test, speaker_ids, y_test in iter_data(opt.test_jsons, vocab, speaker_vocab):
            if opt.speaker_vec_size > 0:
                y_pred = test(X_test, sent_mask_test, speaker_ids)
            else:
                y_pred = test(X_test, sent_mask_test)
            to_segment(id_, y_pred, jsonl_file)
    logger.info('Test took %.3fs', time.time() - start_time)
    sys.exit(0)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--word_vec_path')
    arg_parser.add_argument('--train_word_vec', action='store_true')

    arg_parser.add_argument('--speaker_vocab_path')

    arg_parser.add_argument('--train_jsons', nargs='+')
    arg_parser.add_argument('--valid_jsons', nargs='+')
    arg_parser.add_argument('--test_jsons', nargs='+')
    arg_parser.add_argument('--segment_jsonl')

    arg_parser.add_argument('--model_path')
    arg_parser.add_argument('--load_model', action='store_true')

    arg_parser.add_argument('--speaker_vec_size', type=int)
    arg_parser.add_argument('--dropout', type=float)
    arg_parser.add_argument('--grad_clip', type=int)
    arg_parser.add_argument('--sent_vec_size', type=int)
    arg_parser.add_argument('--sent_backwards', action='store_true')
    arg_parser.add_argument('--hidden_vec_size', type=int)

    arg_parser.add_argument('--learning_rate', type=float)
    arg_parser.add_argument('--epoch', type=int)
    arg_parser.add_argument('--max_epoch_no_improvement', type=int)
    arg_parser.add_argument('--min_improvement', type=float)

    arg_parser.add_argument('--log_path')
    opt = arg_parser.parse_args()

    with codecs.open('logging.yaml', encoding='utf-8') as f:
        log_conf = yaml.load(f)
        log_conf['handlers']['file']['filename'] = opt.log_path
    logging.config.dictConfig(log_conf)

    opt_print = dict(vars(opt))
    opt_print['train_jsons'] = len(opt_print['train_jsons'])
    opt_print['valid_jsons'] = len(opt_print['valid_jsons'])
    opt_print['test_jsons'] = len(opt_print['test_jsons'])
    logger.info(opt_print)

    try:
        main(opt)
    except Exception as e:
        logger.exception('Error')
