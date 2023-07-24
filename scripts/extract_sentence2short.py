"""
抽取长句主干，变为短句
"""

import os
import json
import urllib
from pyltp import Segmentor, Postagger, Parser
os.chdir('./DeepMet_chinese')
LTP_DATA_DIR = 'pyltp/ltp_data_v3.4.0'
segmentor_with_vocab = Segmentor(os.path.join(LTP_DATA_DIR, 'cws.model'),
                                 lexicon_path='./corpus4labelling/mydict.txt')
postagger = Postagger(os.path.join(LTP_DATA_DIR, 'pos.model'))
parser = Parser(os.path.join(LTP_DATA_DIR, 'parser.model'))

def arcs2json(arcs):
    arcs_json = []
    for i, (head, relation) in enumerate(arcs):
        if relation == 'HED':
            word = {'dep': words_with_vocab[i], 'gov': 'ROOT', 'pos': 'HED'}
        else:
            word = {'dep': words_with_vocab[i], 'gov': words_with_vocab[head-1], 'pos': relation}
        arcs_json.append(word)
    return arcs_json
class Sentence(object):
    def __init__(self, arcs):
        self.arcs = arcs

    def getHED(self):
        root = None
        for word in self.arcs:
            if word['gov'] == 'ROOT':
                root = word['dep']
        return root

    def getWord(self, HED, wType):
        sbv = None
        for word in self.arcs:
            if word['pos'] == wType and word['gov'] == HED:
                sbv = word['dep']
        return sbv

    def getFirstNotNone(self, array):
        for word in array:
            if word is not None:
                return word
        return None

    def getMain(self):
        re = ''
        hed = self.getHED()
        if hed is not None:
            sbv = self.getWord(hed, 'SBV')  # 主语
            vob = self.getWord(hed, 'VOB')  # 宾语
            fob = self.getWord(hed, 'FOB')  # 后置宾语

            adv = self.getWord(hed, 'ADV')  # 定中
            pob = self.getWord(adv, 'POB')  # 介宾如果没有主语可做主语

            zhuWord = self.getFirstNotNone([sbv, pob])  # 最终主语
            weiWord = hed  # 最终谓语
            binWord = self.getFirstNotNone([vob, fob, pob])  # 最终宾语
            re = '{}{}{}'.format(zhuWord, weiWord, binWord)
        return re.replace('None', '')

sentence = '总理那高大的形象总是浮现在我眼前'
words_with_vocab = segmentor_with_vocab.segment(sentence)
postags = postagger.postag(words_with_vocab)
arcs = parser.parse(words_with_vocab, postags)
# print("\t".join("%d:%s" % (head, relation) for (head, relation) in arcs))

arcs_json = arcs2json(arcs)
example = Sentence(arcs_json)
example_main = example.getMain()

segmentor_with_vocab.release()
postagger.release()
parser.release()
