"""读取get_meta_sub_BERT的输出文件，按照knowledge based方式
找到隐喻动词的平替词的备选词表
Env: 本地, DeepMet"""
import os
import re
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import synonyms
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
def get_syns_synonyms(word):
        """从synonyms工具包获得word的近义词"""
        syn_words, _ = synonyms.nearby(word, 10)
        return keep_only_CN_char(syn_words)

def load_HIT_synonyms(file_path):
    """加载哈工大同义词词林扩展版，并输出同义词集合和相关词集合"""
    sym_words, sym_class_words = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            items = line.strip().split(' ')
            index = items[0]
            if (index[-1] == '='):  # 同义词
                sym_words.append(items[1:])
            if (index[-1] == '#'):  # 近义词
                sym_class_words.append(items[1:])
    return sym_words, sym_class_words

def get_syns_HIT(word, syns_dict):
    """根据哈工大词林找word的同义词"""
    word_syns = []  # 初始化一个词的同义词列表
    for sym_words in syns_dict:  # 遍历同义词表，sym_words为其中的一条
        if word in sym_words:  # 如果句子中的词在同义词表某一条目中，将该条目中它的同义词添加到该词的同义词列表中
            same_sym_words = [w for w in sym_words if w != word]
            word_syns.extend(same_sym_words)
    return keep_only_CN_char(word_syns)

# 10个词
def get_syns_hownet(word):
    """使用Hownet获得特定词汇的近义词，一个sense对应一组近义词。
    Hownet是根据义原（sememe）来确定近义词的，不同sense可能有相同的义原，因此不同sense的近义词可能是一样的"""
    syn_words = hownet_dict_advanced.get_nearest_words(word, language='zh', K=10)
    # 把该词汇对应所有sense的近义词合并，并去除重复项
    syn_words = list(set(flattern_lst_of_lst(syn_words.values())))
    return keep_only_CN_char(syn_words)

def get_same_frame_word(word):
    """输出跟word相同frame的其他词汇"""
    out_words = []
    if word in CFN_word2frame['word'].values:  # 如果word在CFN中存在
        word_frames = CFN_word2frame[CFN_word2frame['word'] == word]['frame']
        out_words = [CFN_frame2word[frame] for frame in word_frames]
        out_words = flattern_lst_of_lst(out_words)
    return keep_only_CN_char(out_words)

if __name__ == '__main__':
    # 读取文件
    filename = 'poetry0-24859w_predict薛24867_edited'
    df_expand = pd.read_excel('./corpus/poetry/' + filename + '_meta_wobert.xlsx')
    # df_expand = pd.read_csv('./corpus/CCL_PSUCMC_meta_bert.csv', converters={"meta_words": ast.literal_eval})
    # df = pd.read_excel('./corpus/poetry/poetry0-100007w_predict薛24867_edited_top1000.xlsx')
    meta_word_col = 'meta_words'  # 找到隐喻词汇所在列的名称
    sentence_col_name = 'content'  # 找到完整句子所在列的名称

    chinese_re = re.compile('[\u4e00-\u9fa5]+')
    meta_filter = [row[meta_word_col] != '' for idx, row in df_expand.iterrows()]
    # # ----------------------用synonyms找隐喻词汇的近义词---------------------------
    df_expand.loc[meta_filter, 'meta_subs_synonyms'] = df_expand.loc[meta_filter, meta_word_col].apply(get_syns_synonyms)
    # # ----------------------用哈工大词林找隐喻词汇的近义词-----------------------------------------------
    HIT_cilin_file_path = './corpus/哈工大词林扩展版/cilin_ex.txt'
    sym_words_list, sym_class_words_list = load_HIT_synonyms(HIT_cilin_file_path)
    df_expand.loc[meta_filter, 'meta_subs_HIT'] = df_expand.loc[meta_filter, meta_word_col].apply(lambda x: get_syns_HIT(x, sym_words_list))
    # # ----------------------用Hownet找隐喻词汇的近义词-----------------------------------------------
    # hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)
    # OpenHowNet.download()
    # df_expand['meta_subs_hownet'] = ''
    # df_expand_copy = df_expand.copy()
    # for idx, row in tqdm(df_expand_copy.iterrows()):
    #     if row[meta_word_col] != '':  #
    #         df_expand.at[idx, 'meta_subs_hownet'] = get_syns_hownet(row[meta_word_col])

    # df_expand.loc[meta_filter, 'meta_subs_hownet'] = df_expand.loc[meta_filter, meta_word_col].apply(get_syns_hownet)
    # ----------------------用中文FrameNet找隐喻词汇的同域词-----------------------------------------------
    with open('./CFN_LEX_CN/CFN_LEX_CN.json') as f:
        CFN_frame2word = json.load(f)
    CFN_word2frame = pd.read_csv('./CFN_LEX_CN/CFN_LEX_CN.csv')
    df_expand.loc[meta_filter, 'meta_subs_CFN'] = df_expand.loc[meta_filter, meta_word_col].apply(get_same_frame_word)


    # ----------------------集合以上所有词汇到一起-----------------------------------------------
    # df_expand['meta_subs'] = df_expand['meta_subs_synonyms'] + \
    #     df_expand['meta_subs_HIT'] + df_expand['meta_subs_hownet'] + \
    #     df_expand['meta_subs_CFN']
    # df_expand.loc[meta_filter, 'meta_subs'] = df_expand.loc[meta_filter, 'meta_subs'].apply(lambda x: list(set(x)))
    df_expand_meta = df_expand[meta_filter]
    df_expand_meta.drop(['predict'], axis=1, inplace=True)
    df_expand_meta.to_excel('./corpus/poetry/' + filename + '_meta_nohownet.xlsx', index=False)


