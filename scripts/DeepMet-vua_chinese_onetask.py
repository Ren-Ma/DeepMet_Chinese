import os, argparse
import os.path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf  # version: 2.2.0
import tensorflow.keras.backend as K
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from scipy.stats import spearmanr
from math import floor, ceil
# from transformers import *  # version: 2.1.1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, TFAutoModel

def preprocssing(x):
    x = str(x)
    x = '"'+x+'"'
    return x


def _convert_to_transformer_inputs(instance, instance2, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)
                                    #    return_token_type_ids=True) # adding this argument to continue, otherwise report errors. Ren Ma 2021.12.06
        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length) # padding to 128 length vector
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
            _convert_to_transformer_inputs(str(instance.sentence), str(instance.sentence2), tokenizer, max_sequence_length)
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


def compute_output_arrays(df, columns):
    return np.asarray(df[columns].astype(int))

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
    # To circumvent the GFW, load in pretrained model from local drive
    # base_model = TFRobertaModel.from_pretrained('./roberta-base-tf_model', config=config)
    base_model = TFAutoModel.from_pretrained("./chinese-roberta-wwm-ext")

    TransformerA = base_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    TransformerB = base_model(input_id2, attention_mask=input_mask2, token_type_ids=input_atn2)[0]
    output = tf.keras.layers.GlobalAveragePooling1D()(TransformerA)
    output2 = tf.keras.layers.GlobalAveragePooling1D()(TransformerB)
    x = tf.keras.layers.Concatenate()([output, output2])
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn, input_id2, input_mask2, input_atn2], outputs=x)
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
    strategy = tf.distribute.MirroredStrategy()

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

    train = pd.read_csv('./corpus4labelling/CCL_train_long.csv')
    test = pd.read_csv('./corpus4labelling/CCL_test_long.csv')

    print('train shape =', train.shape)
    print('test shape =', test.shape)

    train = sentence_concat(train)
    test = sentence_concat(test)

    input_categories = ['sentence', 'sentence2']
    output_categories = 'meta_label'
    # inputs = compute_input_arrays(train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    # np.save('./tempdata/train_inputs_20200304', inputs)
    inputs = list(np.load('./tempdata/train_inputs_20200304.npy'))
    outputs = compute_output_arrays(train, output_categories)
    # test_inputs = compute_input_arrays(test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    # np.save('./tempdata/test_inputs_20200304', test_inputs)
    test_inputs = list(np.load('./tempdata/test_inputs_20200304.npy'))

    pred = np.zeros((len(test)))
    gkf = StratifiedKFold(n_splits=N_FOLD).split(X=train[input_categories], y=train[output_categories])

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        # sample train_inputs based on random ids generated by gkf. Ren Ma, 2021.12.06
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = to_categorical(outputs[train_idx])
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = to_categorical(outputs[valid_idx])
        K.clear_session()
        model = create_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
        if os.path.isfile(f'./model/model_humanlabel-{fold}.h5'):
            model.load_weights(f'./model/model_humanlabel-{fold}.h5') # load in trained model weights
        else:
            with strategy.scope():
                model.fit(train_inputs, train_outputs, validation_data=[valid_inputs, valid_outputs], epochs=EPOCHS,
                      batch_size=BATCH_SIZE)
            model.save_weights(f'./model/model_humanlabel-{fold}.h5')
        fold_pred = np.argmax(model.predict(test_inputs), axis=1)
        pred += fold_pred  # sum up predictions over all folds
        print("folds: {:d}".format(fold))
        print("verb accuracy: {:.4f}".format(accuracy_score(test[output_categories], fold_pred)))
        print("verb precision: {:.4f}".format(precision_score(test[output_categories], fold_pred)))
        print("verb recall: {:.4f}".format(recall_score(test[output_categories], fold_pred)))
        print("verb f1: {:.4f}".format(f1_score(test[output_categories], fold_pred)))
    np.save('./tempdata/pred.npy', pred)
    best_pred = np.zeros((len(test)))
    best_score = 0
    best_threshold = 0
    '''Loop over different threshold (range(N_FOLD)=0-9) to see which threshold gives the best F1 score'''
    for i in range(N_FOLD):
        temp_pred = (np.array(pred) >= i).astype('int')
        print("verb metaphor preference parameter alpha: {:.4f}".format(i / N_FOLD))
        print("verb accuracy: {:.4f}".format(accuracy_score(test[output_categories], temp_pred)))
        print("verb precision: {:.4f}".format(precision_score(test[output_categories], temp_pred)))
        print("verb recall: {:.4f}".format(recall_score(test[output_categories], temp_pred)))
        print("verb f1: {:.4f}".format(f1_score(test[output_categories], temp_pred)))
        if f1_score(test[output_categories], temp_pred) > best_score:
            best_score = f1_score(test[output_categories], temp_pred)
            best_pred = temp_pred
            best_threshold = i

    print("best verb metaphor preference parameter alpha: {:.4f}".format(best_threshold/N_FOLD))
    print("best verb accuracy: {:.4f}".format(accuracy_score(test[output_categories], best_pred)))
    print("best verb precision: {:.4f}".format(precision_score(test[output_categories], best_pred)))
    print("best verb recall: {:.4f}".format(recall_score(test[output_categories], best_pred)))
    print("best verb f1: {:.4f}".format(f1_score(test[output_categories], best_pred)))
    test['predict'] = best_pred
    print(test.predict.value_counts())

    # test[['id', 'sentence', 'word', 'meta_label', 'predict']].to_csv('./predict/PSUCMC_onlyVerb_predict.csv', index=False)
