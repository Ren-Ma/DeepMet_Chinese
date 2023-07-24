"""读取实习生标记的隐喻动词语料，找到隐喻动词的平替词的备选词表，让实习生从中挑选"""
import os
import re
import jieba
import jieba.posseg as pseg
import pandas as pd
from tqdm import tqdm
import ast

os.getcwd()
import numpy as np
# from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array

def keep_only_CN_char(lst):
    """过滤所有非中文汉字"""
    return [w for w in lst if chinese_re.findall(w) != []]

def flattern_lst_of_lst(lst):
    """展开2层嵌套式list"""
    lst = [x for nest_lst in lst for x in nest_lst]  # 方法1
    # lst = sum(lst, [])  # 方法2
    # lst = reduce(lambda x, y: x+y, lst)  # 方法3
    return lst

def word_encode(word):
    """encode a single word to token_ids, without CLS or SEP"""
    token_ids, _ = tokenizer.encode(word)  # encode每个词
    token_ids = token_ids[1: -1]  # 去掉前缀CLS和后缀SEP
    return token_ids


def sentence_encode(sentence):
    sentence_tokenized = jieba.lcut(sentence)
    sentence_token_ids = [word_encode(word) for word in sentence_tokenized]
    return sentence_token_ids


def split_sentence(sentence, split_word):
    """按照split_word把sentence分割开，输出为list"""
    split_word_re = '(' + split_word + ')'  # 把re表达式放进括号内，表示在输出中保留分割项。
    sentence_split = re.split(split_word_re, sentence)
    sentence_split = [x for x in sentence_split if x != '']
    return sentence_split


def predict_replaced_word(list_of_token_ids):
    """根据token_ids来预测mask掉的词，输出前100个候选词"""
    list_of_token_ids = flattern_lst_of_lst(list_of_token_ids)
    # add CLS and SEP tokens before and after. NOTE!!! the '[]' out of list_of_token_ids is required
    token_ids = to_array([[101] + list_of_token_ids + [102]])
    segment_ids = np.zeros_like(token_ids)
    # NOTE!!! the [0] at the end
    probas = model.predict([token_ids, segment_ids])[0]
    # NOTE!!! the [0] after token_ids, and after *103]
    this_replaced_word = [tokenizer._token_dict_inv[x] for x in probas[token_ids[0] == 103][0].argsort()[-100:][::-1]]
    return keep_only_CN_char(this_replaced_word)


def predict_wobert_one_row(row, sentence_col_name, meta_word_col):
    """基于wobert预测隐喻词汇的平替词"""
    sentence_split = split_sentence(row[sentence_col_name], (row[meta_word_col]))
    meta_words_ids = [word_encode(row[meta_word_col])]  # 获取meta_words的token_ids

    # [[sent1_word1_ids, sent1_word2_ids, ...], [sent2_word1_ids, sent2_word2_ids, ...], ...]
    sentence_token_ids = [sentence_encode(sent) for sent in sentence_split]
    # [sent1_word1_ids, sent1_word2_ids, ..., sent2_word1_ids, sent2_word2_ids, ...]
    sentence_token_ids = flattern_lst_of_lst(sentence_token_ids)

    sentence_token_ids_masked = [ids if ids not in meta_words_ids else [tokenizer._token_mask_id] for ids in
                                 sentence_token_ids]
    predicted_meta_words_wobert = predict_replaced_word(sentence_token_ids_masked)
    return predicted_meta_words_wobert


def add_words2dict(df, meta_word_col):
    """把df中一列的词汇添加到jieba分词的mydict.txt中"""
    old_dict = open('./corpus/mydict.txt', encoding='utf8').read().splitlines()
    words = [word for word in df[meta_word_col] if len(word) > 0]
    old_dict.extend(words)
    new_dict = pd.DataFrame({'word': old_dict})
    new_dict = new_dict['word'].drop_duplicates()
    return new_dict


if __name__ == '__main__':
    # 读取文件
    df = pd.read_csv('./corpus/CCL_PSUCMC.csv', converters={"meta_words": ast.literal_eval})
    # df = pd.read_excel('./corpus/poetry/poetry0-100007w_predict薛24867_edited_top1000.xlsx')
    meta_word_col = 'meta_words'  # 找到隐喻词汇所在列的名称
    sentence_col_name = 'sentence'  # 找到完整句子所在列的名称
    # df[meta_word_col] = df[meta_word_col].fillna('')
    # df[meta_word_col] = df[meta_word_col].apply(lambda x: list(set(x.split('、'))))  # 分割隐喻词并去除重复词
    df[meta_word_col] = df[meta_word_col].apply(lambda x: list(set(x)))
    df_expand = df.explode(meta_word_col, ignore_index=True)  # 把meta_word_col这一列展开，其他列重复
    # 把隐喻词汇添加到jieba分词的自定义词典
    # new_dict = add_words2dict(df, meta_word_col)
    # new_dict.to_csv('./corpus/mydict0331.txt', index=False, header=False)
    jieba.load_userdict("./corpus/mydict0331.txt")
    # --------------------------------------用wobert预测隐喻词汇的平替词-----------------------------------------------
    # 启动模型
    config_path = './chinese_wobert_plus_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = './chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = './chinese_wobert_plus_L-12_H-768_A-12/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                    with_mlm=True)  # 建立模型，加载权重

    chinese_re = re.compile('[\u4e00-\u9fa5]+')
    df_expand['meta_sub_wobert'] = ''
    df_expand_copy = df_expand.copy()
    # 采用apply功能，只输出df中的隐喻句子
    meta_filter = [row[meta_word_col] != '' for idx, row in df_expand.iterrows()]
    # df_expand_onlymeta = df_expand[df[meta_word_col] != ['']].apply(predict_wobert_one_row, axis=1)
    # 采用loop迭代，输出整个df
    for idx, row in tqdm(df_expand.iterrows()):
        if row[meta_word_col] != '':  # 如果没有分割项，则输出原句
            df_expand_copy.at[idx, 'meta_sub_wobert'] = predict_wobert_one_row(row, sentence_col_name,
                                                                                           meta_word_col)
    num_syns = 10  # 指定要获取的近义词数目
    # --------------------------------------用synonyms找隐喻词汇的近义词-----------------------------------------------
    import synonyms

    def get_syns_synonyms(word):
        syn_words, _ = synonyms.nearby(word, num_syns)
        return keep_only_CN_char(syn_words)
    # --------------------------------------用哈工大词林找隐喻词汇的近义词-----------------------------------------------
    def load_synonyms(file_path):
        """加载哈工大同义词词林扩展版，并输出同义词集合和相关词集合"""
        sym_words, sym_class_words = [], []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                items = line.strip().split(' ')
                index = items[0]
                if (index[-1] == '='):
                    sym_words.append(items[1:])
                if (index[-1] == '#'):
                    sym_class_words.append(items[1:])
        return sym_words, sym_class_words

    def get_syns_HIT(word):
        sym_words_list_copy = sym_words_list.copy()
        word_syns = []  # 初始化一个词的同义词列表
        for sym_words in sym_words_list_copy:  # 遍历同义词表，sym_words为其中的一条
            if word in sym_words:  # 如果句子中的词在同义词表某一条目中，将该条目中它的同义词添加到该词的同义词列表中
                sym_words.remove(word)
                word_syns.extend(sym_words)
        return keep_only_CN_char(word_syns)

    synonyms_file_path = './corpus/哈工大词林扩展版/cilin_ex.txt'
    sym_words_list, sym_class_words_list = load_synonyms(synonyms_file_path)

    # --------------------------------------用Hownet找隐喻词汇的近义词-----------------------------------------------
    import OpenHowNet
    hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)
    # OpenHowNet.download()

    def get_syns_hownet(word):
        """使用Hownet获得特定词汇的num_syns个近义词，一个sense对应一组近义词。
        Hownet是根据义原（sememe）来确定近义词的，不同sense可能有相同的义原，因此不同sense的近义词可能是一样的"""
        syn_words = hownet_dict_advanced.get_nearest_words(word, language='zh', K=num_syns)
        # 把该词汇对应所有sense的近义词合并，并去除重复项
        syn_words = list(set(flattern_lst_of_lst(syn_words.values())))
        return keep_only_CN_char(syn_words)

    # --------------------------------------用中文FrameNet找隐喻词汇的同域词-----------------------------------------------
    import  json
    with open('./CFN_LEX_CN/CFN_LEX_CN.json') as f:
        CFN_frame2word = json.load(f)
    CFN_word2frame = pd.read_excel('./CFN_LEX_CN/CFN_LEX_CN.xlsx')

    def get_same_frame_word(word):
        """输出跟word相同frame的其他词汇"""
        out_words = []
        if word in CFN_word2frame['word'].values:  # 如果word在CFN中存在
            word_frames = CFN_word2frame[CFN_word2frame['word'] == word]['frame']
            out_words = [CFN_frame2word[frame] for frame in word_frames]
            out_words = flattern_lst_of_lst(out_words)
        return keep_only_CN_char(out_words)


    df_expand.loc[meta_filter, 'meta_subs_synonyms'] = df_expand.loc[meta_filter, meta_word_col].apply(get_syns_synonyms)
    df_expand.loc[meta_filter, 'meta_subs_HIT'] = df_expand.loc[meta_filter, meta_word_col].apply(get_syns_HIT)
    df_expand.loc[meta_filter, 'meta_subs_hownet'] = df_expand.loc[meta_filter, meta_word_col].apply(get_syns_hownet)
    df_expand.loc[meta_filter, 'meta_subs_CFN'] = df_expand.loc[meta_filter, meta_word_col].apply(get_same_frame_word)
    df_expand['meta_subs'] = df_expand['meta_subs_synonyms'] + \
                             df_expand['meta_subs_HIT'] + \
                             df_expand['meta_subs_hownet'] + \
                             df_expand['meta_subs_CFN']
    df_expand.loc[meta_filter, 'meta_subs'] = df_expand.loc[meta_filter, 'meta_subs'].apply(lambda x: list(set(x)))
    df_expand_meta = df_expand[meta_filter]
    df_expand_meta.to_csv('./corpus/poetry/df_expand_meta.csv')



