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

def long_to_wide(df):
    '''    convert long format dataframe to wide format dataframe, so
    to predict by word_bert model    '''
    ids, word, local, sentence, token_ids,  = [], [], [], [], []
    replaced_word, sent_bf_mask, sent_aft_mask = [], [], []
    for idx in tqdm(df['id'].unique()[:1000]):
        metaphor_verb_MetaEq1_subset = metaphor_verb_MetaEq1[metaphor_verb_MetaEq1.id == idx]
        word_idx = 0
        for _,row in metaphor_verb_MetaEq1_subset.iterrows():
            if row['word'] != row['sentence'][word_idx:word_idx+len(row['word'])]:
                print(row['id'], row['word'], '<<|>>', row['sentence'][word_idx:word_idx+len(row['word'])])
            if row['metaphor'] == 1:
                this_token_ids, this_replaced_word, this_sent_bf_mask, this_sent_aft_mask= process_row(row, word_idx)
                ids.append(row['id']), word.append(row['word'])
                local.append(row['local']), sentence.append(row['sentence'])
                # word_pos.append(this_word_pos)
                token_ids.append(this_token_ids), replaced_word.append(this_replaced_word)
                sent_bf_mask.append(this_sent_bf_mask), sent_aft_mask.append(this_sent_aft_mask)
            word_idx += len(row['word'])  # 每循环一行，把当前word的index移到相应位置

    return pd.DataFrame({"id": ids, "word": word, "local": local, "sentence": sentence,
                         # "word_pos": word_pos,
                         'token_ids': token_ids, 'replaced_word': replaced_word,
                         'sent_bf_mask': sent_bf_mask, 'sent_aft_mask': sent_aft_mask})

def process_row(row, word_idx):
    this_token_ids, _ = tokenizer.encode(row['sentence'])  # 输出token_ids和segment_ids
    # this_word_pos = [0] * len(row['sentence'])  # 重置one_id_word_pos
    # this_word_pos[word_idx: word_idx+len(row['word'])] = [1] * len(row['word'])  # 更改相应位置one_id_word_pos值
    this_token_ids[word_idx + 1: word_idx + len(row['word'])+1] = [tokenizer._token_mask_id]  # 把隐喻词（可能有>=1个id）替换为一个mask
    this_token_ids = to_array([this_token_ids])
    this_segment_ids = np.zeros_like(this_token_ids)
    probas = model.predict([this_token_ids, this_segment_ids])[0]
    this_replaced_word = [tokenizer._token_dict_inv[x] for x in probas[word_idx+1].argsort()[-50:][::-1]]
    this_sent_bf_mask = row['sentence'][:word_idx]
    this_sent_aft_mask = row['sentence'][word_idx+len(row['word']):]
    return this_token_ids, this_replaced_word, this_sent_bf_mask, this_sent_aft_mask

meta_verb4replace = long_to_wide(metaphor_verb_MetaEq1)
meta_verb4replace_noPunc = meta_verb4replace[meta_verb4replace['word'].apply(lambda x: chinese_re.findall(x) != [])]
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

# 用mlm模型预测被mask掉的部分
for _, row in tqdm(meta_verb4replace_noPunc.iterrows()):
    token_ids = to_array([row['token_ids']])
    probas = model.predict([token_ids, np.zeros_like(token_ids)])[0]
    # print(tokenizer.decode(probas[3:5].argmax(axis=1)))  # 结果正是“技术”
    print(tokenizer.decode(probas.argmax(axis=1)))
print([tokenizer._token_dict_inv[x] for x in probas[3].argsort()[-10:][::-1]])
