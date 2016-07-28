#!/usr/bin/env python
# coding: utf-8
import json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return vars(obj)
        except TypeError:
            pass
        return json.JSONEncoder.default(self, obj)


class Sent(object):
    def __init__(self, start, end, speaker, text):
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = text

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and \
               self.speaker == other.speaker and self.text == other.text


class Topic(object):
    def __init__(self, start, end, description, subtopics):
        self.start = start
        self.end = end
        self.description = description
        self.subtopics = subtopics

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and \
               self.description == other.description and \
               self.subtopics == other.subtopics


class Segment(object):
    def __init__(self, id_, boundary_indices, gap_count):
        self.id = id_
        self.boundaryIndices = boundary_indices
        self.gapCount = gap_count


def append_to_jsonl(obj, jsonl):
    json.dump(obj, jsonl, ensure_ascii=False, encoding='utf-8',
              cls=CustomEncoder)
    jsonl.write('\n')


def pred_to_segment_jsonl(ids, boundary_labels, jsonl, boundary_label=1):
    for id_, y_a_doc in zip(ids, boundary_labels):
        boundary_indices = [i for i, v in enumerate(y_a_doc)
                            if v == boundary_label]
        segment = Segment(id_, boundary_indices, len(y_a_doc))
        append_to_jsonl(segment, jsonl)


def to_json(id_, sents, topics, out_file):
    """
    Args:
        id_ (str)
        sents ([dict])
        topics ([dict])
        out_file (writable)
    """
    json.dump(
        {
            'id': id_,
            'topicCount': len(topics),
            'topics': topics,
            'sentCount': len(sents),
            'sents': sents
        },
        out_file,
        ensure_ascii=False,
        indent=4,
        cls=CustomEncoder
    )
