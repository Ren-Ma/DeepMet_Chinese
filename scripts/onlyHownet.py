"""读取实习生标记的隐喻动词语料，找到隐喻动词的平替词的备选词表，让实习生从中挑选"""
import os
import re
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import OpenHowNet
import json

os.getcwd()

def keep_only_CN_char(lst):
    """过滤所有非中文汉字"""
    return [w for w in lst if chinese_re.findall(w) != []]

def flattern_lst_of_lst(lst):
    """展开2层嵌套式list"""
    lst = [x for nest_lst in lst for x in nest_lst]  # 方法1
    # lst = sum(lst, [])  # 方法2
    # lst = reduce(lambda x, y: x+y, lst)  # 方法3
    return lst


# 10个词
def get_syns_hownet(word):
    """使用Hownet获得特定词汇的近义词，一个sense对应一组近义词。
    Hownet是根据义原（sememe）来确定近义词的，不同sense可能有相同的义原，因此不同sense的近义词可能是一样的"""
    syn_words = hownet_dict_advanced.get_nearest_words(word, language='zh', K=10)
    # 把该词汇对应所有sense的近义词合并，并去除重复项
    syn_words = list(set(flattern_lst_of_lst(syn_words.values())))
    return keep_only_CN_char(syn_words)


if __name__ == '__main__':
    # 读取文件
    df_expand = pd.read_csv('./corpus/CCL_PSUCMC_meta_bert.csv')
    # df_expand = pd.read_csv('./corpus/CCL_PSUCMC_meta_bert.csv', converters={"meta_words": ast.literal_eval})
    # df = pd.read_excel('./corpus/poetry/poetry0-100007w_predict薛24867_edited_top1000.xlsx')
    meta_word_col = 'meta_words'  # 找到隐喻词汇所在列的名称
    sentence_col_name = 'sentence'  # 找到完整句子所在列的名称

    chinese_re = re.compile('[\u4e00-\u9fa5]+')
    meta_filter = [row[meta_word_col] != '' for idx, row in df_expand.iterrows()]
    hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)
    # OpenHowNet.download()
    df_expand['meta_subs_hownet'] = ''
    df_expand_copy = df_expand.copy()
    for idx, row in tqdm(df_expand_copy.iterrows()):
        if row[meta_word_col] != '':  #
            df_expand.at[idx, 'meta_subs_hownet'] = get_syns_hownet(row[meta_word_col])


    df_expand_meta = df_expand[meta_filter]
    df_expand_meta.to_csv('./corpus/CCL_PSUCMC_meta_sub_hownet.csv')



