import os
import re
import jieba
import jieba.posseg as pseg
import pandas as pd
from tqdm import tqdm
os.getcwd()
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array


# 仅查找中文汉字，过滤掉标点符号
chinese_re = re.compile('[\u4e00-\u9fa5]+')

# CCL_2018_predict_onlyVerb = pd.read_csv('./predict/CCL_2018_predict_onlyVerb.csv')
# PSUCMC_train_features = pd.read_csv('./RuiHe_Chinese_Metaphor_Detection-main/Data/PSUCMC_train_features_onlyVerb.csv')
# PSUCMC_test_features = pd.read_csv('./RuiHe_Chinese_Metaphor_Detection-main/Data/PSUCMC_test_features_onlyVerb.csv')
# PSUCMC_onlyVerb = pd.concat([PSUCMC_train_features, PSUCMC_test_features], ignore_index = True)
# data1 = CCL_2018_predict_onlyVerb.rename({'predict':'metaphor'}, axis = 1)
# data2 = PSUCMC_onlyVerb.rename({'label':'metaphor'}, axis = 1)
# data1.drop(['pos', 'label'], axis = 1, inplace=True)
# data2.drop(['pos', 'newID'], axis = 1, inplace=True)
# metaphor_verb = pd.concat([data1, data2], ignore_index = True)
# metaphor_verb_wide = metaphor_verb.groupby(['id']).max()
# MetaEq1_idxs = metaphor_verb_wide[metaphor_verb_wide['metaphor'] == 1].index
# metaphor_verb_MetaEq1 = metaphor_verb[metaphor_verb.id.isin(MetaEq1_idxs)]

metaphor_verb_MetaEq1 = pd.read_csv('metaphor_verb_MetaEq1.csv')
config_path = './chinese_wobert_plus_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_wobert_plus_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)  # 建立模型，加载权重

def word_encode(word):
    """encode a single word without CLS or SEP"""
    token_ids, _ = tokenizer.encode(word)  # encode每个词
    token_ids = token_ids[1: -1]  # 去掉前缀CLS和后缀SEP
    return token_ids

def dataset_word_encode(df):
    """encode one column of words in a dataframe"""
    df['token_ids'] = df['word'].apply(lambda word: word_encode(word))
    # df.loc[df['metaphor'] == 1, 'token_ids'] == [tokenizer._token_mask_id]  # mask the metaphor words token ids
    return df

def this_sent_long_to_wide(df):
    """convert long format to wide, concat items as list"""
    meta_list = df.groupby(['id'])['metaphor'].apply(list)[0]  #
    word_list = df.groupby(['id'])['word'].apply(list)[0]  #
    token_ids_list = df.groupby(['id'])['token_ids'].apply(list)[0]  #
    return meta_list, word_list, token_ids_list

def predict_replaced_word(list_of_token_ids):
    """predict masked word, return top50 candidates"""
    list_of_token_ids = [x for nest_lst in list_of_token_ids for x in nest_lst]
    # add CLS and SEP tokens before and after. NOTE!!! the '[]' out of list_of_token_ids is required
    token_ids = to_array([[101] + list_of_token_ids + [102]])
    segment_ids = np.zeros_like(token_ids)
    # NOTE!!! the [0] at the end
    probas = model.predict([token_ids, segment_ids])[0]
    # NOTE!!! the [0] after token_ids, and after *103]
    this_replaced_word = [tokenizer._token_dict_inv[x] for x in probas[token_ids[0] == 103][0].argsort()[-50:][::-1]]
    return this_replaced_word

def long_to_wide(df):
    '''    convert long format dataframe to wide format dataframe, so
    to predict by word_bert model    '''
    df = dataset_word_encode(df)
    ids, word, local, sentence, token_ids,  = [], [], [], [], []
    replaced_word, sent_bf_mask, sent_aft_mask = [], [], []
    for idx in tqdm(df['id'].unique()):
        metaphor_verb_MetaEq1_subset = metaphor_verb_MetaEq1[metaphor_verb_MetaEq1.id == idx]
        meta_list, word_list, token_ids_list = this_sent_long_to_wide(metaphor_verb_MetaEq1_subset)
        for meta_pos in np.where(np.array(meta_list) == 1)[0]:  # go to the position where metaphor equals 1
            token_ids_list_masked = token_ids_list.copy()
            token_ids_list_masked[meta_pos] = [tokenizer._token_mask_id]
            this_replaced_word = predict_replaced_word(token_ids_list_masked)
            this_sent_bf_mask = ''.join(word_list[:meta_pos])
            this_sent_aft_mask = ''.join(word_list[meta_pos+1:])

            ids.append(idx)
            word.append(word_list[meta_pos])
            local.append(metaphor_verb_MetaEq1_subset.iloc[meta_pos]['local'])
            sentence.append(metaphor_verb_MetaEq1_subset.iloc[meta_pos]['sentence'])
            replaced_word.append(this_replaced_word)
            token_ids.append(token_ids_list_masked)
            sent_bf_mask.append(this_sent_bf_mask)
            sent_aft_mask.append(this_sent_aft_mask)

    return pd.DataFrame({"id": ids, "word": word, "local": local, "sentence": sentence,
                         'token_ids': token_ids, 'replaced_word': replaced_word,
                         'sent_bf_mask': sent_bf_mask, 'sent_aft_mask': sent_aft_mask})

if __name__ == '__main__':
    meta_verb4replace = long_to_wide(metaphor_verb_MetaEq1)
    meta_verb4replace_noPunc = meta_verb4replace[meta_verb4replace['word'].apply(lambda x: chinese_re.findall(x) != [])]
    meta_verb4replace_noPunc.to_csv('meta_verb4replace_20220209.csv')

# meta_verb4replace_noPunc.to_csv('meta_verb4replace_noPunc.csv', index = False)
# meta_verb4replace_noPunc = pd.read_csv('meta_verb4replace_noPunc.csv',
#                                        dtype = {'token_ids':'list'})
# sentence = '门一开，人们就涌入了图书馆'
# token_ids, segment_ids = tokenizer.encode(u'科学技术是第一生产力')
# token_ids, segment_ids = tokenizer.encode(u'这辆汽车很耗油')
# token_ids, segment_ids = tokenizer.encode(u'门一开，人们就涌入了图书馆')
# mask掉“技术”
# token_ids[3] = token_ids[4] = tokenizer._token_mask_id
# token_ids[1]  = tokenizer._token_mask_id
# token_ids, segment_ids = to_array([token_ids], [segment_ids])

# # 用mlm模型预测被mask掉的部分
# for _, row in tqdm(meta_verb4replace_noPunc.iterrows()):
#     token_ids = to_array([row['token_ids']])
#     probas = model.predict([token_ids, np.zeros_like(token_ids)])[0]
#     # print(tokenizer.decode(probas[3:5].argmax(axis=1)))  # 结果正是“技术”
#     print(tokenizer.decode(probas.argmax(axis=1)))
# print([tokenizer._token_dict_inv[x] for x in probas[3].argsort()[-10:][::-1]])
