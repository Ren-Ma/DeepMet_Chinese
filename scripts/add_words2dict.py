import jieba
import pandas as pd
from datetime import date
# 把隐喻词汇添加到jieba分词的自定义词典

def add_words2dict(df, word_col):
    """把df中一列的词汇添加到jieba分词的mydict.txt中"""
    old_dict = open(current_dict_path, encoding='utf8').read().splitlines()
    words = [word for word in df[word_col] if len(word) > 0]
    old_dict.extend(words)
    new_dict = pd.DataFrame({'word': old_dict})
    new_dict = new_dict['word'].drop_duplicates()
    return new_dict

today = date.today().strftime("%m%d")
new_dict_path = './corpus/mydict/mydict' + today + '.txt'

# 每次换文件需要改动的变量
current_dict_path = './corpus/mydict/mydict0405.txt'  # 当前的dictionary
filename = 'poetry0-100007w_predict薛24867_edited'
df = pd.read_excel('./corpus/poetry/' + filename + '.xlsx')
word_col = 'meta_words'  # 词汇所在列的名称
sentence_col = 'content'  # 句子所在列的名称

df[word_col] = df[word_col].fillna('')
df[word_col] = df[word_col].apply(lambda x: list(set(x.split('、'))))  # 分割隐喻词并去除重复词
df_expand = df.explode(word_col, ignore_index=True)  # 把meta_word_col这一列展开，其他列重复
new_dict = add_words2dict(df_expand, word_col)
new_dict.to_csv(new_dict_path, index=False, header=False)