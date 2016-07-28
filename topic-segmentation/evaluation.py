#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import argparse
import codecs
import json
import numpy as np
import segeval
from sklearn.metrics import make_scorer as sk_make_scorer
import sys


def boundary_indices_to_labels(gap_count, boundary_indices):
    """
    Parameter
    ---------
    gap_count : int
    boundary_indices : [int]

    Returns
    -------
    boundary_labels : [int]

    >>> boundary_indices_to_labels(5, [1, 3])
    [0, 1, 0, 1, 0]
    >>> boundary_indices_to_labels(5, [])
    [0, 0, 0, 0, 0]
    >>> boundary_indices_to_labels(5, [4])
    [0, 0, 0, 0, 1]
    """
    labels = [0 for i in xrange(gap_count)]
    for i in boundary_indices:
        labels[i] = 1
    return labels


def boundary_indices_to_masses(gap_count, boundary_indices):
    """
    >>> boundary_indices_to_masses(12, [3, 10])
    [4, 7, 2]
    >>> boundary_indices_to_masses(12, [])
    [13]
    >>> boundary_indices_to_masses(12, [11])
    [12, 1]
    """
    masses = []
    previous = -1
    for i in boundary_indices:
        masses.append(i - previous)
        previous = i
    masses.append(gap_count - previous)
    return masses


def boundary_labels_to_masses(labels, boundary_label=1):
    """
    Parameter
    ---------
    labels : [int or str]
        label[i] = 1 when gap[i] is a boundary, = 0 otherwise
    boundary_label : int or str

    Returns
    -------
    masses : [int]

    >>> boundary_labels_to_masses([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    [4, 7, 2]
    >>> boundary_labels_to_masses([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    [13]
    >>> boundary_labels_to_masses([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    [12, 1]
    """
    labels_ = labels.tolist() if isinstance(labels, np.ndarray) else labels
    masses = []
    count = 0
    for label in labels_:
        count += 1
        if label == boundary_label:
            masses.append(count)
            count = 0
    masses.append(count + 1)
    return masses


def window_diff(ref, pred, punish_zero_seg=False, boundary_label=1):
    """
    Parameter
    ---------
    ref : [int or str]
        ref boundary labels
    pred : [int or str]
        pred boundary labels
    punish_zero_seg : bool
        if True zero-segmentation will get 1.0
    boundary_label : int or str

    >>> window_diff([-1, -1, 1, -1, -1], [-1, -1, -1, 1, -1])
    0.5
    >>> window_diff([-1, -1, 1, -1, -1], [-1, -1, -1, -1, -1])
    0.5
    >>> window_diff([-1, -1, 1, -1, -1], [-1, -1, -1, -1, -1], punish_zero_seg=True)
    1.0
    """
    masses_ref = boundary_labels_to_masses(ref, boundary_label=boundary_label)
    masses_pred = boundary_labels_to_masses(pred, boundary_label=boundary_label)
    if punish_zero_seg and len(masses_pred) == 1:
        return 1.0
    return float(segeval.window_diff(masses_pred, masses_ref))


def average_window_diff(refs, preds, punish_zero_seg=False, boundary_label=1):
    """
    Parameter
    ---------
    refs : [[int]]
        refs[i][j] is reference label of j-th gap in i-th doc
    preds : [[int]]
        preds[i][j] is predicted label of j-th gap in i-th doc
    punish_zero_seg : bool
        if True zero-segmentation will get 1.0

    Returns
    -------
    wds : [float]
        window_diffs for all docs
    average : float
    """
    window_diffs = []
    for ref, pred in zip(refs, preds):
        window_diffs.append(window_diff(
            ref, pred, punish_zero_seg, boundary_label=boundary_label))
    average = sum(window_diffs) / len(window_diffs)
    return window_diffs, average


def pk(ref, pred, punish_zero_seg=False, boundary_label=1):
    masses_ref = boundary_labels_to_masses(ref, boundary_label=boundary_label)
    masses_pred = boundary_labels_to_masses(pred, boundary_label=boundary_label)
    if punish_zero_seg and len(masses_pred) == 1:
        return 1.0
    return float(segeval.pk(masses_pred, masses_ref))


def average_pk(refs, preds, punish_zero_seg=False, boundary_label=1):
    """
    Parameter
    ---------
    refs : [[int]]
        refs[i][j] is reference label of j-th gap in i-th doc
    preds : [[int]]
        preds[i][j] is predicted label of j-th gap in i-th doc

    Returns
    -------
    pks : [float]
        pks for all docs
    average : float
    """
    pks = []
    for ref, pred in zip(refs, preds):
        pks.append(pk(ref, pred, punish_zero_seg, boundary_label=boundary_label))
    average = sum(pks) / len(pks)
    return pks, average


def _window_diff_func(y_ref, y_pred, punish_zero_seg=False, boundary_label=1):
    _, average_wd = average_window_diff(
            y_ref, y_pred, punish_zero_seg=punish_zero_seg, boundary_label=boundary_label)
    return average_wd


def window_diff_scorer(punish_zero_seg=False, boundary_label=1):
    scorer = sk_make_scorer(_window_diff_func, greater_is_better=False,
                            punish_zero_seg=punish_zero_seg,
                            boundary_label=boundary_label)
    return scorer


def read_segment(segment_file):
    segment_dict = {}
    for line in segment_file:
        segment_json = json.loads(line)
        labels = boundary_indices_to_labels(
                segment_json['gapCount'], segment_json['boundaryIndices'])
        segment_dict[segment_json['id']] = labels
    return segment_dict


def main(opt):
    with codecs.open(opt.ref_jsonl, encoding='utf-8') as ref_file:
        ref_dict = read_segment(ref_file)
    for pred_jsonl in opt.pred_jsonl:
        with codecs.open(pred_jsonl, encoding='utf-8') as pred_file:
            pred_dict = read_segment(pred_file)
        pk_list = []
        wd_list = []
        print('==> {} <=='.format(pred_jsonl))
        for doc_name, pred_a_doc in pred_dict.iteritems():
            if doc_name not in ref_dict:
                continue
            ref_a_doc = ref_dict[doc_name]
            pk_ = pk(ref_a_doc, pred_a_doc)
            wd = window_diff(ref_a_doc, pred_a_doc)
            print('name:{} Pk:{:.3f} WD:{:.3f}'.format(doc_name, pk_, wd))
            pk_list.append(pk_)
            wd_list.append(wd)
        print('Pk(average):{:.3f} WD(average):{:.3f}'.format(
            sum(pk_list) / len(pk_list), sum(wd_list) / len(wd_list)))

    sys.exit(0)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('ref_jsonl')
    arg_parser.add_argument('pred_jsonl', nargs='+')
    opt = arg_parser.parse_args()
    main(opt)
