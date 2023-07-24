## 把CCL_PSUCMC句子处理成long格式

import re
import jieba
import jieba.posseg as pseg
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import ast

os.getcwd()

jieba.enable_paddle()  # 启动paddle模式
jieba.load_userdict("mydict.txt")
CCL_PSUCMC = pd.read_csv('CCL_PSUCMC.csv', converters={"meta_word": ast.literal_eval})


# 中文分句
def cut_sent(para):
    para = re.sub('([，。！：；？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)   # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def cut_one_sent_append(row, sent, local_sent, longdata, meta_label):
    sent_words = pseg.cut(sent, use_paddle=True)
    for word, flag in sent_words:
        longdata['word'].append(word)
        longdata['tag'].append(flag)
        longdata['id'].append(row.id)
        longdata['sentence'].append(row.sentence)
        longdata['local'].append(local_sent)
        longdata['meta_label'].append(meta_label)
    return longdata

def tokenize_tagging(df):
    longdata = {'id': [], 'sentence': [], 'word': [], 'tag': [], 'local': [], 'meta_label': []}
    for _,row in tqdm(df.iterrows()):
        local_sents = cut_sent(row['sentence'])
        if row.meta_word == ['']:
            for local_sent in local_sents:
                longdata = cut_one_sent_append(row, local_sent, local_sent, longdata, False)
        elif len(row.meta_word) == 1:  # 只有一个meta词汇
            for local_sent in local_sents:
                local_sent_split = re.split('('+row['meta_word'][0]+')', local_sent)  # split并且保留分割项
                for splitted_sent in local_sent_split:
                    longdata = cut_one_sent_append(row, splitted_sent, local_sent, longdata, splitted_sent == row['meta_word'][0])
                # local_sents = [x for nest_lst in local_sents for x in nest_lst]
        else:  # 有多个meta词汇
            split_re = '|'.join(row.meta_word)
            for local_sent in local_sents:
                local_sent_split = re.split('('+split_re+')', local_sent)  # split并且保留分割项
                for splitted_sent in local_sent_split:
                    longdata = cut_one_sent_append(row, splitted_sent, local_sent, longdata, splitted_sent in row['meta_word'])
    return pd.DataFrame(longdata)


CCL_PSUCMC_long = tokenize_tagging(CCL_PSUCMC)
CCL_PSUCMC_long.to_csv('CCL_PSUCMC_long.csv', index=False)


