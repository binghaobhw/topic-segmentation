#!/usr/bin/env python
# coding: utf-8
import argparse
import codecs
import os
import xml.etree.ElementTree as et
import re
import sys

from json_core import Sent, Topic, to_json


NAMESPACE_NITE = 'http://nite.sourceforge.net/'
OTHER_DESCRIPTION = 'other_description'
NITE_ID_ATTR = '{' + NAMESPACE_NITE + '}id'
NAME_ATTR = 'name'
TOPICNAME_TAG = 'topicname'
NITE_POINTER_TAG = '{' + NAMESPACE_NITE + '}pointer'
NITE_CHILD_TAG = '{' + NAMESPACE_NITE + '}child'
TOPIC_TAG = 'topic'
HREF_ATTR = 'href'
W_TAG = 'w'
START_TIME_ATTR = 'starttime'
END_TIME_ATTR = 'endtime'
TRUNC_ATTR = 'trunc'
PUNC_ATTR = 'punc'


def get_default_topics(root):
    default_topics = {}
    # recursively
    for topicname_element in root.iter(TOPICNAME_TAG):
        id_ = topicname_element.get(NITE_ID_ATTR)
        default_topics[id_] = topicname_element.get(NAME_ATTR)
    return default_topics


def get_speaker_words(root):
    nite_id_attr = root.get(NITE_ID_ATTR)
    speaker = re.search(r'\.(.+)\.', nite_id_attr).group(1)
    words = [e for e in root]
    return speaker, words


def topics_lines(topics_root, default_topics, speaker_segment_dict, speaker_word_dict):
    topics = []
    sents = []
    speaker_segment_pos_dict = {}
    for speaker in speaker_segment_dict:
        speaker_segment_pos_dict[speaker] = 0
    for topic_element in topics_root:
        topic = parse_ami_topic(topic_element, default_topics,
                                speaker_segment_dict,
                                speaker_segment_pos_dict,
                                speaker_word_dict, sents)
        topics.append(topic)
    return topics, sents


def is_valid_word_element(e):
    return e.tag == W_TAG and TRUNC_ATTR not in e.attrib


def get_valid_word_elements(e_list):
    return filter(is_valid_word_element, e_list)


def words_to_sent(speaker, words):
    if not words:
        return None
    text = ' '.join(map(lambda x: x.text, words))
    if not text:
        return None
    start_time = words[0].get(START_TIME_ATTR)
    end_time = words[-1].get(END_TIME_ATTR)
    return Sent(start_time, end_time, speaker, text)


def get_speaker_segments(root):
    nite_id_attr = root.get(NITE_ID_ATTR)
    speaker = re.search(r'\.(.+)\.', nite_id_attr).group(1)
    segments = []
    for segment_element in root:
        child_element = segment_element.find(NITE_CHILD_TAG)
        href_attr = child_element.get(HREF_ATTR)
        _, range_ = href_attr.split('#')
        groups = re.findall(r'words(\d+)', range_)
        if len(groups) == 1:
            start = int(groups[0])
            end = start
        else:
            start, end = map(int, groups)
        segments.append((start, end))
    return speaker, segments


def parse_ami_topic(topic_element, default_topics, speaker_segment_dict,
                    speaker_segment_pos_dict, speaker_word_dict, sents):
    subtopics = []
    description = ''
    topic_start = len(sents)
    next_line_num = topic_start
    if OTHER_DESCRIPTION in topic_element.attrib:
        description = topic_element.get(OTHER_DESCRIPTION)

    for element in topic_element:
        if element.tag == TOPIC_TAG:
            subtopic = parse_ami_topic(element, default_topics,
                                       speaker_segment_dict,
                                       speaker_segment_pos_dict,
                                       speaker_word_dict, sents)
            subtopics.append(subtopic)
        elif element.tag == NITE_POINTER_TAG and not description:
            href_attr = element.get(HREF_ATTR)
            topic_id = href_attr[href_attr.rfind('top'):-1]
            description = default_topics[topic_id]
        elif element.tag == NITE_CHILD_TAG:
            href_attr = element.get(HREF_ATTR)
            filename, range_ = href_attr.split('#')
            speaker = re.search(r'\.(\w)\.', filename).group(1)
            groups = re.findall(r'words(\d+)', range_)
            if len(groups) == 1:
                child_start = int(groups[0])
                child_end = child_start
            else:
                child_start, child_end = map(int, groups)
            segments = speaker_segment_dict[speaker]
            segment_pos = speaker_segment_pos_dict[speaker]
            words = speaker_word_dict[speaker]
            start = child_start
            for pos in xrange(segment_pos, len(segments)):
                segment_start, segment_end = segments[pos]
                end = segment_end if segment_end <= child_end else child_end
                sent = words_to_sent(speaker, get_valid_word_elements(words[start:end + 1]))
                start = end + 1
                if sent:
                    sents.append(sent)
                    next_line_num += 1
                if segment_end >= child_end:
                    if segment_end == child_end:
                        segment_pos = pos + 1
                    break
            speaker_segment_pos_dict[speaker] = segment_pos
    topic_end = len(sents) - 1
    return Topic(topic_start, topic_end, description, subtopics)


def parse_word(word_xmls):
    speaker_word_dict = {}
    for word_xml in word_xmls:
        root = et.parse(word_xml).getroot()
        speaker, words = get_speaker_words(root)
        speaker_word_dict[speaker] = words
    return speaker_word_dict


def parse_segment(segment_xmls):
    speaker_segment_dict = {}
    for segment_xml in segment_xmls:
        root = et.parse(segment_xml).getroot()
        speaker, segments = get_speaker_segments(root)
        speaker_segment_dict[speaker] = segments
    return speaker_segment_dict


def ami_to_json(meeting_id, topic_xml, word_xmls, segment_xmls, default_topics_xml, out_file):
    speaker_word_dict = parse_word(word_xmls)
    speaker_segment_dict = parse_segment(segment_xmls)
    default_topics_root = et.parse(default_topics_xml).getroot()
    default_topics = get_default_topics(default_topics_root)
    topic_root = et.parse(topic_xml).getroot()
    topics, sents = topics_lines(topic_root, default_topics, speaker_segment_dict, speaker_word_dict)
    to_json(meeting_id, sents, topics, out_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('topic_dir')
    arg_parser.add_argument('word_dir')
    arg_parser.add_argument('segment_dir')
    arg_parser.add_argument('default_topics_xml')
    arg_parser.add_argument('output_dir')
    args = arg_parser.parse_args()

    topic_xmls = map(lambda x: os.path.join(args.topic_dir, x),
                     os.listdir(args.topic_dir))
    all_word_xmls = map(lambda x: os.path.join(args.word_dir, x),
                        os.listdir(args.word_dir))
    all_segment_xmls = map(lambda x: os.path.join(args.segment_dir, x),
                           os.listdir(args.segment_dir))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for topic_xml in topic_xmls:
        meeting_id = re.search(r'^([\w\d]+)',
                               os.path.split(topic_xml)[1]).group(1)
        word_xmls = filter(lambda x: meeting_id in x, all_word_xmls)
        segment_xmls = filter(lambda x: meeting_id in x, all_segment_xmls)
        output_path = os.path.join(args.output_dir,
                                   '{}.json'.format(meeting_id))
        with codecs.open(output_path, mode='w', encoding='utf-8') as output:
            ami_to_json(meeting_id, topic_xml, word_xmls, segment_xmls, args.default_topics_xml, output)
    sys.exit(0)
