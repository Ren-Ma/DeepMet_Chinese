import os
import re
import json
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split
from statistics import mode
os.getcwd()
# os.chdir('./DeepMet_chinese')
from utils_DeepMet import flattern_lst_of_lst, keep_only_CN_char
from unlp import UTextSimilarity
textsim_model = UTextSimilarity('../chinese-roberta-wwm-ext', 'sentbert')


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

def optimal_CFN(word, word_frames, sentence):
    """把word替换为frame，计算与原句的相似度，选相似度最大的frame"""
    sim_scores = []
    for frame in tqdm(word_frames):
        sim_scores_frame = []
        words_same_frame = CFN_word2frame.loc[CFN_word2frame.CFN == frame, 'word']
        for word_same_frame in words_same_frame:
            replaced_sent = re.sub(word, word_same_frame, sentence)
            sim_scores_frame.append(textsim_model.run(sentence, replaced_sent))
        sim_scores.append(sum(sim_scores_frame)/len(sim_scores_frame))  # 把平均值append到sim_scores里
    max_sim_idx2 = sim_scores.index(max(sim_scores))
    optim_frame_words_sub = word_frames.iloc[max_sim_idx2]
    return optim_frame_words_sub

def get_CFN(word, sentence):
    """输出word的frame"""

    if word not in CFN_word2frame['word'].values:  # 如果这个词汇找不到对应的frame的话,利用同义词的概念域
        syn_words = get_syns_HIT(word, sym_words_list)
        # 只保留有CFN库里可以找到的同义词
        syn_words = [syn_word for syn_word in syn_words if syn_word in CFN_word2frame['word'].values]
        if syn_words == []:  # 如果没有同义词的话
            word_frame = '未知域'
        else:
            syn_frames = [get_CFN(syn_word, sentence) for syn_word in syn_words]
            syn_frames = pd.Series(list(set(syn_frames)))
            word_frame = optimal_CFN(word, pd.Series(syn_frames), sentence)
    else:   # 如果word在CFN中存在
        word_frames = CFN_word2frame[CFN_word2frame['word'] == word]['CFN']
        if len(word_frames) == 1:
            word_frame = word_frames.iloc[0]
        else:
            word_frame = optimal_CFN(word, word_frames, sentence)  # 现在只选第一个
    return word_frame

def get_json_format(row):
    """把文本转为run_translation_bart_chinese.py所需的json格式"""
    print('处理到第'+str(row['index'])+'行了')
    row['meta_word_CFN'] = get_CFN(row[meta_word_col], row[sentence_col])
    row['sent_src'] = re.sub(row[meta_word_col], row[meta_sub_col], row[sentence_col])
    row['meta_sub_CFN'] = get_CFN(row[meta_sub_col],  row['sent_src'])
    # meta_sub = '<V>' + row[meta_sub_col] + '<V>'
    meta_sub = '<V>' + row[meta_sub_col] + ':' + row['meta_sub_CFN'] + '<V>'
    row['sent_src_ftr'] = row['meta_word_CFN'] + '<EOT>' + re.sub(row[meta_word_col], meta_sub, row[sentence_col])
    # row['sent_src'] = re.sub(row[meta_word_col], meta_sub, row[sentence_col])
    row['data_json'] = {"translation": {"source": row['sent_src_ftr'],
                                        "target": row[sentence_col]}}
    return row

def write2json(df, col_name, dataset_type):
    """dataset_type: train, valid, test"""
    train_data = open('./corpus/CCL_PSUCMC平替词/'+dataset_type+'.json', 'w', encoding='utf-8')
    for line in df[col_name]:
        train_data.write(str(line) + '\n')
        # train_data.write(line)
    train_data.close()

if __name__ == '__main__':
    chinese_re = re.compile('[\u4e00-\u9fa5]+')

    # 读取文件
    # df_expand = pd.read_csv('./corpus/CCL_PSUCMC_meta_bert.csv', converters={"meta_words": ast.literal_eval})
    # filename = 'CCL_PSUCMC_meta_sub_3000-5639_4fairseq_test'
    # df = pd.read_excel('./corpus/CCL_PSUCMC平替词/' + filename + '.xlsx')
    # df1 = pd.read_excel('./corpus/CCL_PSUCMC平替词/CCL_PSUCMC_meta_sub_1500薛长梅.xlsx')
    # df2 = pd.read_excel('./corpus/CCL_PSUCMC平替词/CCL_PSUCMC_meta_sub_1500-3000徐凯悦.xlsx')
    # df3 = pd.read_excel('./corpus/CCL_PSUCMC平替词/CCL_PSUCMC_meta_sub_3000-5639陈国政.xlsx')
    # df4 = pd.read_excel('./corpus/CCL_PSUCMC平替词/汪曾祺小说经典-4.17_meta_sub.xlsx')
    # df5 = pd.read_excel('./corpus/CCL_PSUCMC平替词/汪曾祺散文-4.17_meta_sub.xlsx')
    # df6 = pd.read_excel('./corpus/CCL_PSUCMC平替词/老舍2W_meta_nohownet.xlsx')
    # df7 = pd.read_excel('./corpus/CCL_PSUCMC平替词/阿城小说集-陈国政_meta_sub.xlsx')
    # df8 = pd.read_excel('./corpus/CCL_PSUCMC平替词/阿城随笔集-陈国政_meta_sub.xlsx')
    # df9 = pd.read_excel('./corpus/CCL_PSUCMC平替词/poetry0-24859w_predict薛24867_edited_meta_nohownet.xlsx')

    # df1 = pd.read_excel('./corpus/CCL_PSUCMC平替词/df标注前1400条.xlsx')
    # df2 = pd.read_excel('./corpus/CCL_PSUCMC平替词/df徐凯悦.xlsx')
    # df3 = pd.read_excel('./corpus/CCL_PSUCMC平替词/df薛长梅.xlsx')
    # df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], ignore_index=True)
    # df = pd.concat([df1, df2, df3], ignore_index=True)
    meta_word_col = 'meta_word'  # 隐喻词汇所在列的名称
    sentence_col = 'sentence'  # 完整句子所在列的名称
    meta_sub_col = 'meta_sub_new'
    HIT_cilin_file_path = '../corpus/哈工大词林扩展版/cilin_ex.txt'
    sym_words_list, sym_class_words_list = load_HIT_synonyms(HIT_cilin_file_path)

    CFN_word2frame = pd.read_csv('../CFN_LEX_CN/CFN_LEX_CN.csv')
    # jieba.load_userdict("./corpus/mydict/mydict0408.txt")  # 每次都要用到最新的mydict
    # df = df[df[meta_sub_col].notna()]
    # df[meta_sub_col] = df[meta_sub_col].apply(lambda x: list(set(x.split('、'))))
    # df_expand = df.explode(meta_sub_col, ignore_index=True)  # 把meta_sub_col这一列展开，其他列重复
    # df = df.apply(lambda x: get_source_and_target(x), axis=1)  #
    df_expand = pd.read_csv('./corpus/CCL_PSUCMC平替词/df_expand1.csv')
    row = df_expand.iloc[28]
    row = get_json_format(row)
    df_expand_json = df_expand.apply(get_json_format, axis=1)  #
    df_expand_json.to_csv('./corpus/CCL_PSUCMC平替词/df_expand_json4.csv', index=False)
    # df[meta_sub_col] = df[meta_sub_col].fillna('')
    # write_out_data(df, nsrc='source', ntgt='target')
    # train_valid, test = train_test_split(df_expand_json, test_size=0.1) # 10% test
    # train, valid = train_test_split(train_valid, test_size=0.2) # 90% * 20% valid
    # train.to_excel('./corpus/CCL_PSUCMC平替词/train.xlsx', index=False)
    # valid.to_excel('./corpus/CCL_PSUCMC平替词/valid.xlsx', index=False)
    # test.to_excel('./corpus/CCL_PSUCMC平替词/test.xlsx', index=False)
    # write2json(train, 'data_json', 'train')
    # write2json(valid, 'data_json', 'valid')
    # write2json(test, 'data_json', 'test')
    #
    # write2json(train_json, 'data_json', 'train_no_frame')
    # write2json(valid_json, 'data_json', 'valid_no_frame')
    # write2json(test_json, 'data_json', 'test_no_frame')
    # 在VS code（）中打开json文件把单引号替换为双引号，否则run_translation_chinese_bart.py会报错。

# 往bert词汇表中添加自定义词汇，并resize预训练模型权重
# from transformers import BartForConditionalGeneration, BertTokenizer
# model_path = '/Users/ma/Desktop/PLM/fnlp_bart-large-chinese_added_tokens'
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BartForConditionalGeneration.from_pretrained(model_path)
# new_tokens = ['<EOT>', '<V>', 'No_Frame']
# new_tokens.extend(CFN_word2frame['frame'].unique().tolist())
# num_added_toks = tokenizer.add_tokens(new_tokens)  # 返回一个数，表示加入的新词数量
# model.resize_token_embeddings(len(tokenizer))
# tokenizer.save_pretrained(model_path)  # 还是保存到原来的model文件夹下，这时候文件夹下多了三个文件



