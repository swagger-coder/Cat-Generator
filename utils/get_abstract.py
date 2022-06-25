#-*- encoding:utf-8 -*-
from __future__ import print_function
import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import sys
try:
    # reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

def get_abstract(content):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=content, lower=True, source = 'all_filters')
    abstract_str = []
    i = 0
    for item in tr4s.get_key_sentences(num=3):
        # abstract_str += "生成的第{}个摘要为：{}\n".format(i+1, item.sentence)
        abstract_str.append(item.sentence)
        i += 1
    return abstract_str