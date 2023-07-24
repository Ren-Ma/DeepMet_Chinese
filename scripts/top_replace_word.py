import os
import re
import pandas as pd
from tqdm import tqdm
os.getcwd()
import numpy as np
from ast import literal_eval
from unlp import UTextSimilarity
textsim_model = UTextSimilarity('./chinese-roberta-wwm-ext', 'sentbert')

def calc_sim_one_row(row):
    sim_scores = []
    for word in row['replaced_word']:
        sim_scores.append(textsim_model.run(row['sentence'], row['sent_bf_mask']+word+row['sent_aft_mask']))
    max_sim_idx = sim_scores.index(max(sim_scores))
    row['max_sim_idx'] = max_sim_idx
    row['top_candic_word'] = row['replaced_word'][max_sim_idx]
    row['sent_word_replaced'] = row['sent_bf_mask'] + '<<' + row['top_candic_word'] + '>>' + row['sent_aft_mask']
    return row

def calc_sim_df(df):
    meta_verb4replace['sent_bf_mask'] = meta_verb4replace['sent_bf_mask'].fillna('')
    meta_verb4replace['sent_aft_mask'] = meta_verb4replace['sent_aft_mask'].fillna('')
    colnames = df.columns.values.tolist()
    colnames.extend(['max_sim_idx', 'top_candic_word', 'sent_word_replaced'])
    out_df = pd.DataFrame(columns=colnames)
    for _, row in tqdm(df.iterrows()):
        row = calc_sim_one_row(row)
        out_df = out_df.append(row)
    return out_df

if __name__ == '__main__':
    meta_verb4replace = pd.read_csv('meta_verb4replace_20220209.csv', converters={"replaced_word": literal_eval})
    meta_verb4replace_new = calc_sim_df(meta_verb4replace)
    meta_verb4replace_new.to_csv('meta_verb4replace_topword.csv', index = False)