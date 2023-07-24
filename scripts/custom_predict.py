import re
import jieba
import jieba.posseg as pseg
import os, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf  # version: 2.2.0
import tensorflow.keras.backend as K
from transformers import AutoTokenizer, TFAutoModel
import time
chinese_re = re.compile('[\u4e00-\u9fa5]+')

jieba.enable_paddle()  # 启动paddle模式
jieba.load_userdict("./corpus4labelling/mydict.txt")

# 中文分句
def cut_sent(para):
    para = re.sub('([，。！：；？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，
    #把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 破折号、英文双引号等忽略，
    return para.split("\n")

def one_sent_to_DeepMet_format(input_sent):
    sentences, words, tags, local_sentences = [], [], [], []
    local_sents = cut_sent(input_sent)  # 先断成短句
    for local_sent in local_sents:
        local_sent_words = pseg.cut(local_sent, use_paddle=True)  # jieba分词+paddle模式
        for word, flag in local_sent_words:
            sentences.append(input_sent)
            words.append(word)
            tags.append(flag)
            local_sentences.append(local_sent)
    return pd.DataFrame({"sentence": sentences,
                         "word": words,
                         "tag": tags,
                         "local": local_sentences})

def df_to_DeepMet_format(df, col_name):
    cut_sent_compile = re.compile('([，。！：；？\?])|(\.{6})|(\…{2})|([。！？\?][”’])')
    ids, sentences, words, tags, local_sentences = [], [], [], [], []
    for _, row in tqdm(df.iterrows()):
        local_sents = cut_sent(row[col_name])
        for local_sent in local_sents:
            local_sent_words = pseg.cut(local_sent, use_paddle=True)
            for word, flag in local_sent_words:
                ids.append(row['id'])
                sentences.append(row[col_name])
                words.append(word)
                tags.append(flag)
                local_sentences.append(local_sent)
    return pd.DataFrame({"id": ids, "word": words, "tag": tags,
                         "local": local_sentences, "sentence": sentences})

def preprocssing(x):
    return '"' + str(x) + '"'

def _convert_to_transformer_inputs(instance, instance2, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)
        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)  # padding to 128 length vector
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        return [input_ids, input_masks, input_segments]

    """compute and transform the global feature. Ren Ma, 2021.12.02"""
    input_ids, input_masks, input_segments = return_id(
        instance, None, 'longest_first', max_sequence_length)

    """compute and transform the local feature. Ren Ma, 2021.12.02"""
    input_ids2, input_masks2, input_segments2 = return_id(
        instance2, None, 'longest_first', max_sequence_length)
    return [input_ids, input_masks, input_segments,
            input_ids2, input_masks2, input_segments2]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    input_ids2, input_masks2, input_segments2 = [], [], []
    """df[columns] = ['id', 'sentence', 'word', 'label', 'tag', 'local', 'sentence2']"""
    for _, instance in tqdm(df[columns].iterrows()):
        ids, masks, segments, ids2, masks2, segments2 = \
            _convert_to_transformer_inputs(str(instance.sentence), str(instance.sentence2), tokenizer,
                                           max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        input_ids2.append(ids2)
        input_masks2.append(masks2)
        input_segments2.append(segments2)
    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32),
            np.asarray(input_ids2, dtype=np.int32),
            np.asarray(input_masks2, dtype=np.int32),
            np.asarray(input_segments2, dtype=np.int32)]

# Siamese structure
def create_model():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_id2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn2 = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    # config = RobertaConfig.from_pretrained('roberta-base')
    # config.output_hidden_states = False
    # base_model = TFRobertaModel.from_pretrained('roberta-base', config=config)
    # base_model = TFRobertaModel.from_pretrained('./roberta-base-tf_model', config=config)
    base_model = TFAutoModel.from_pretrained("./chinese-roberta-wwm-ext")

    TransformerA = base_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    TransformerB = base_model(input_id2, attention_mask=input_mask2, token_type_ids=input_atn2)[0]
    output = tf.keras.layers.GlobalAveragePooling1D()(TransformerA)
    output2 = tf.keras.layers.GlobalAveragePooling1D()(TransformerB)
    x = tf.keras.layers.Concatenate()([output, output2])
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn, input_id2, input_mask2, input_atn2],
                                  outputs=x)
    return model

def sentence_concat(df):
    df['sentence'] = df.sentence.apply(lambda x: preprocssing(x)) \
                           + "[SEP]" + df.word.apply(lambda x: preprocssing(x)) \
                           + "[SEP]" + df.tag.apply(lambda x: preprocssing(x)) \
                           + "[SEP]" + df.tag.apply(lambda x: preprocssing(x))
    df['sentence2'] = df.local.apply(lambda x: preprocssing(x)) \
                              + "[SEP]" + df.word.apply(lambda x: preprocssing(x)) \
                              + "[SEP]" + df.tag.apply(lambda x: preprocssing(x)) \
                              + "[SEP]" + df.tag.apply(lambda x: preprocssing(x))
    return df

if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Experimental environment: Titan RTX
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sequence_length", default=128, type=int, required=False)
    parser.add_argument("--hidden_size", default=768, type=int, required=False)
    parser.add_argument("--random_state", default=2020, type=int, required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)
    parser.add_argument("--n_fold", default=10, type=int, required=False)
    # parser.add_argument("--n_fold", default=5, type=int, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--dropout_rate", default=0.2, type=float, required=False)
    parser.add_argument("--validation_split", default=0.1, type=float, required=False)
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False)
    args = parser.parse_args()
    MAX_SEQUENCE_LENGTH = args.max_sequence_length
    HIDDEN_SIZE = args.hidden_size
    RANDOM_STATE = args.random_state
    EPOCHS = args.epochs
    N_FOLD = args.n_fold
    BATCH_SIZE = args.batch_size
    DROPOUT_RATE = args.dropout_rate
    VALIDATION_SPLIT = args.validation_split
    LEARNING_RATE = args.learning_rate

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained("./chinese-roberta-wwm-ext")

    # Input_sent = sys.argv[1]  # input one sentence
    # Input_sent = input()  # input one sentence
    # Input_sent = '教堂的钟声撞击在我的胸口'

    # input_df = one_sent_to_DeepMet_format(Input_sent)

    # meta_train = pd.read_csv('./胡锡进微博/comments_前1十万clean.csv')  #　数据来自'微博数据.ipynb'处理结果
    # meta_train = pd.read_csv('./胡锡进微博/weibo_前0十万clean.csv')  # 数据来自'微博数据.ipynb'处理结果
    # novel = pd.read_excel('./corpus4labelling/poetry0-100007.xlsx')
    # novel['id'] = novel.index
    # novel['sentences'] = novel['sentences'].fillna('')

    poetry = pd.read_excel('./corpus4labelling/poetry200012-300009.xlsx')
    poetry['content'] = poetry['content'].fillna('')

    input_df = df_to_DeepMet_format(poetry, 'content')
    input_df = input_df[input_df.word.apply(lambda x: chinese_re.findall(x) != [])]
    input_df_copy = input_df.copy()

    input_df = sentence_concat(input_df)
    input_categories = ['sentence', 'sentence2']
    strategy = tf.distribute.MirroredStrategy()
    input_df_inputs = compute_input_arrays(input_df, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    # np.save('./tempdata/老舍文集', input_df_inputs)
    # input_df_inputs = list(np.load('./tempdata/老舍文集.npy'))

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
    model.load_weights(f'./model/model_humanlabel-9.h5')  # load in trained model weights, model-2 is the best one
    start_time = time.process_time()
    with strategy.scope():
        input_sent_pred = np.argmax(model.predict(input_df_inputs), axis=1)
    end_time = time.process_time()
    print('prediction process took %f seconds' % (end_time - start_time))

    input_df_copy['predict'] = input_sent_pred
    input_df_wide = input_df_copy.groupby(['id']).max()
    input_df_idxs = input_df_wide[input_df_wide['predict'] > 0].index
    input_df_copy = input_df_copy[input_df_copy.id.isin(input_df_idxs)]
    input_df_copy = input_df_copy[['id', 'predict', 'word', 'local', 'sentence', 'tag']]
    print(input_df_copy.predict.value_counts())
    input_df_copy_meta = input_df_copy[input_df_copy['predict'] == 1]

    input_df_copy_meta_group = input_df_copy_meta.groupby(['id'])['word'].apply(list)
    input_df_copy_meta_group['id'] = input_df_copy_meta_group.index
    novel_m = pd.merge(poetry, input_df_copy_meta_group, how='left', on=['id'])
    novel_m.to_excel('./predict/poetry200012-300009w_predict.xlsx', index=False)
    # input_df_copy_meta.to_excel('./predict/汪曾祺散文—predict.xlsx', index=False)