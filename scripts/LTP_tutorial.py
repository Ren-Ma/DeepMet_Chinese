import os
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
os.chdir('./DeepMet_chinese')
LTP_DATA_DIR = '../pyltp/ltp_data_v3.4.0'
paragraph = '他叫汤姆去拿外衣。'
paragraph = '总理那高大的形象总是浮现在我眼前'
# --------------------- 断句 ------------------------
sentence = SentenceSplitter.split(paragraph)[0]

# -------------------- Context Manager -------------
with Segmentor(os.path.join(LTP_DATA_DIR, "cws.model")) as s:
    words = s.segment(sentence)
    print("\t".join(words))

# --------------------- 分词 ------------------------
segmentor = Segmentor(os.path.join(LTP_DATA_DIR, 'cws.model'))
words = segmentor.segment(sentence)
print('\t'.join(words))

segmentor_with_vocab = Segmentor(os.path.join(LTP_DATA_DIR, 'cws.model'),
                                 lexicon_path='./corpus4labelling/mydict.txt')
words_with_vocab = segmentor_with_vocab.segment(sentence)
print("\t".join(words_with_vocab), "\t\t| With Vocab")

# segmentor_with_force_vocab = Segmentor(os.path.join(LTP_DATA_DIR, "cws.model"),
#                                        force_lexicon_path='lexicon.txt')  # 除上述功能外，原本合并在一起的亦会拆分
# words_with_force_vocab = segmentor_with_force_vocab.segment(sentence)
# print("\t".join(words_with_force_vocab), "\t| Force Vocab")

# --------------------- 词性标注 ------------------------
postagger = Postagger(os.path.join(LTP_DATA_DIR, 'pos.model'))
postags = postagger.postag(words)
# postags = postagger.postag(["中国","进出口","银行","与","中国银行","加强","合作"])
print("\t".join(postags))

# --------------------- 语义依存分析 ------------------------
parser = Parser(os.path.join(LTP_DATA_DIR, 'parser.model'))
arcs = parser.parse(words, postags)
print("\t".join("%d:%s" % (head, relation) for (head, relation) in arcs))

# --------------------- 命名实体识别 ------------------------
recognizer = NamedEntityRecognizer(os.path.join(LTP_DATA_DIR, "ner.model"))
netags = recognizer.recognize(words, postags)
print("\t".join(netags))

# --------------------- 语义角色标注 ------------------------
labeller = SementicRoleLabeller(os.path.join(LTP_DATA_DIR, "pisrl.model"))
roles = labeller.label(words, postags, arcs)
for index, arguments in roles:
    print(index, " ".join(["%s: (%d,%d)" % (name, start, end) for (name, (start, end)) in arguments]))

segmentor.release()
segmentor_with_vocab.release()
# segmentor_with_force_vocab.release()
segmentor.release()
postagger.release()
parser.release()
recognizer.release()
labeller.release()
