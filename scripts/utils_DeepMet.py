import re
import jieba
import jieba.posseg as pseg
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import ast
from sklearn.model_selection import train_test_split

chinese_re = re.compile('[\u4e00-\u9fa5]+')  # 仅查找中文汉字，过滤掉标点符号

def flattern_lst_of_lst(lst):
    """展开2层嵌套式list"""
    lst = [x for nest_lst in lst for x in nest_lst]  # 方法1
    # lst = sum(lst, [])  # 方法2
    # lst = reduce(lambda x, y: x+y, lst)  # 方法3
    return lst

def keep_only_CN_char(lst):
    """过滤所有非中文汉字"""
    return [w for w in lst if chinese_re.findall(w) != []]

def clean_text(x):
    x = re.sub(' ', '', x) # 删除空格
    x = re.sub('０', '0', x)
    x = re.sub('１', '1', x)
    x = re.sub('２', '2', x)
    x = re.sub('３', '3', x)
    x = re.sub('４', '4', x)
    x = re.sub('５', '5', x)
    x = re.sub('６', '6', x)
    x = re.sub('７', '7', x)
    x = re.sub('８', '8', x)
    x = re.sub('９', '9', x)
    x = re.sub('．', '.', x)
    x = re.sub('％', '%', x)
    x = x.replace(u'\xa0', u'') # 删除'\xa0': 不间断空白符
    return x

def cut_sent(para):
    """ 中文分句 """
    para = re.sub('([，。！：；？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，
    #把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 破折号、英文双引号等忽略，
    return para.split("\n")

def jieba_tokenize_add_space(sent):
    """jieba分词，以空格隔开"""
    sent_tok = jieba.lcut(sent)
    sent_tok_add_space = ' '.join(sent_tok)
    return sent_tok_add_space