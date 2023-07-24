import os
import re
import json
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split
os.getcwd()
os.chdir('/data/renma/DeepMet/DeepMet_chinese')

from unlp import UTextSimilarity
textsim_model = UTextSimilarity('./chinese-roberta-wwm-ext', 'sentbert')
CFN_word2frame = pd.read_excel('./CFN_LEX_CN/CFN_LEX_CN.xlsx')
word = "吃"
sentence = "这辆汽车很吃油"
word_frames = CFN_word2frame[CFN_word2frame['word'] == word]['CFN']

def optimal_CFN(word, word_frames, sentence):
    """把word替换为frame，计算与原句的相似度，选相似度最大的frame"""
    sim_scores1, sim_scores2 = [], []
    for frame in tqdm(word_frames):
        # 计算sim_scores1----------------
        replaced_sent = re.sub(word, frame, sentence)
        sim_scores1.append(textsim_model.run(sentence, replaced_sent))
        # 计算sim_scores2----------------
        sim_scores_frame = []
        words_same_frame = CFN_word2frame.loc[CFN_word2frame.CFN == frame, 'word']
        for word_same_frame in words_same_frame:
            replaced_sent = re.sub(word, word_same_frame, sentence)
            sim_scores_frame.append(textsim_model.run(sentence, replaced_sent))
        sim_scores2.append(sum(sim_scores_frame)/len(sim_scores_frame))  # 把平均值append到sim_scores里
    max_sim_idx1 = sim_scores1.index(max(sim_scores1))
    optim_frame_sub = word_frames.iloc[max_sim_idx1]
    max_sim_idx2 = sim_scores2.index(max(sim_scores2))
    optim_frame_words_sub = word_frames.iloc[max_sim_idx2]
    return [optim_frame_sub, optim_frame_words_sub]

def get_CFN(word, sentence):
    """输出word的frame"""
    if word not in CFN_word2frame['word'].values:  # 如果这个词汇找不到对应的frame的话
        word_frame = '未知域'
    else:   # 如果word在CFN中存在
        word_frames = CFN_word2frame[CFN_word2frame['word'] == word]['CFN']
        if len(word_frames) == 1:
            word_frame = word_frames.iloc[0]
        else:
            word_frame = optimal_CFN(word, word_frames, sentence)  # 现在只选第一个
    return word_frame

x = get_CFN(word, sentence)
print(x)

