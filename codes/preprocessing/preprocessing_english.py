# 第一步：加载数据集
dataname = 'yelp_review_polarity_csv (2)'
num_of_sample = str(1000)
with open('d:/pythonwork/newclassification/dataset/' + dataname + '_' + num_of_sample +'.txt', 'r',
          encoding='gb18030', errors='ignore') as a:
    texts = [i.strip().lower() for i in a.readlines()]
with open('d:/pythonwork/newclassification/dataset/' + dataname + '_' + num_of_sample +'_label.txt', 'r') as a:
    labels = [i.strip() for i in a.readlines()]
# with open('e:/pythonwork/newclassification/dataset/' + dataname + num_of_sample +'.txt', 'r', encoding='gb18030', errors='ignore') as a:
#     texts = [i.strip().lower() for i in a.readlines()]
# with open('e:/pythonwork/newclassification/dataset/' + dataname + num_of_sample +'_label.txt', 'r') as a:
#     labels = [i.strip() for i in a.readlines()]

#
# 第二步：扩展缩写词
from contractions import CONTRACTION_MAP
listtemp = list(CONTRACTION_MAP.keys())
for i in listtemp:
    CONTRACTION_MAP[i.upper()] = CONTRACTION_MAP[i].upper()
    CONTRACTION_MAP[i.title()] = CONTRACTION_MAP[i].title()

def expand_contractions(s, contractions):
    # print(contractions)
    for i in contractions.keys():
        s = s.replace(i, contractions[i])
    return s
texts = [expand_contractions(sentence, CONTRACTION_MAP) for sentence in texts]
# print(texts[0])

# 第三步：删除特殊字符
import re
def remove_characters_before_tokenization(sentence, keep_apostrophes=False):
    sentence = sentence.strip()
    # 要标点符号
    if keep_apostrophes:
        PATTERN = r'[?|$|&|*|%|@|(|)|~]'
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    # 不要标点符号
    else:
        PATTERN = r'[^a-zA-Z0-9 ]'
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence
texts = [remove_characters_before_tokenization(sentence, False) for sentence in texts]
# print(texts[0])


#
# # 第四步：分词
from nltk.tokenize import word_tokenize
def tokenize_text(text):
    word_tokens = word_tokenize(text)
    return word_tokens
token_list = [tokenize_text(i) for i in texts]
# print(token_list[0])

# 第五步：去停用词
# 用文本文件中的停用词表
def remove_stopwords(tokens):
    file = 'd:/pythonwork/newclassification/englishstopwords.txt'
    with open(file, 'r') as f:
        stopword_list2 = f.readlines()
    stopword_list2 = [word.strip() for word in stopword_list2]
    filtered_tokens = [token for token in tokens if token not in stopword_list2]
    return filtered_tokens
token_list = [remove_stopwords(sentence) for sentence in token_list]
# print(token_list[0])
#
# 第六步：去除重复字母
from nltk.corpus import wordnet
def remove_replicated_char(text):
    for i in range(len(text)):
        for j in range(len(text[i])):
            if wordnet.synsets(text[i][j]):
                continue
            else:
                p = re.findall(r'([a-zA-Z]*)([a-zA-Z])\2([a-zA-Z]*)', text[i][j])
                # print(p)
                while p != []:
                    text[i][j] = text[i][j].replace(p[0][1], '', 1)
                    if wordnet.synsets(text[i][j]):
                        break
                    p = re.findall(r'([a-zA-Z]*)([a-zA-Z])\2([a-zA-Z]*)', text[i][j])
    return text
token_list = remove_replicated_char(token_list)
# print(token_list[0])
#
#
# 第七步：错误词矫正
import Levenshtein, collections
def tokens(text):
    """
    Get all words from the corpus
    """
    return re.findall('[a-z]+', text.lower())
WORDS = tokens(open('d:/pythonwork/newclassification/big.txt').read())
WORD_COUNTS = collections.Counter(WORDS)
#
# # print(None in WORD_COUNTS)
#
def correct(word):
    if re.match(r'[a-zA-Z]+', word):
        if word in WORD_COUNTS:
            return word
        else:
            candidates = [w for w in WORD_COUNTS if Levenshtein.distance(word, w) == 1] or \
                         [w for w in WORD_COUNTS if Levenshtein.distance(word, w) == 2] or \
                         [word]
            return max(candidates, key=WORD_COUNTS.get)
    else:
        return word

from tqdm import tqdm

token_list = [[correct(j) for j in i] for i in tqdm(token_list)]
# print(token_list[0])
#
# 第八步：进行词性标注和词形还原
from nltk.tag import pos_tag
token_list_tag = [pos_tag(sentence) for sentence in token_list]

from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()
for i in range(len(token_list)):
    for j in range(len(token_list[i])):
        if token_list_tag[i][j][1] == 'JJ' or 'RB':
            token_list[i][j] = wnl.lemmatize(token_list[i][j], 'a')
        if token_list_tag[i][j][1].startswith('N'):
            token_list[i][j] = wnl.lemmatize(token_list[i][j], 'n')
        elif token_list_tag[i][j][1].startswith('V'):
            token_list[i][j] = wnl.lemmatize(token_list[i][j], 'v')
## 第九步：保存数据
tokens_new = []
labels_new = []
for i in range(len(token_list)):
    if len(token_list[i]) > 1:
        tokens_new.append(token_list[i])
        labels_new.append(labels[i])
with open('d:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_' + num_of_sample + '.txt',
          'w', encoding='utf-8') as a:
    for i in range(len(tokens_new)):
        a.write(' '.join(tokens_new[i]) + '\n')
with open('d:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_' + num_of_sample +
          '_label.txt', 'w', encoding='utf-8') as a:
    for i in range(len(labels_new)):
        a.write(labels_new[i] + '\n')

# with open('e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname  + num_of_sample + '.txt', 'w', encoding='utf-8') as a:
#     for i in range(len(tokens_new)):
#         a.write(' '.join(tokens_new[i]) + '\n')
# with open('e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname  + num_of_sample + '_label.txt', 'w', encoding='utf-8') as a:
#     for i in range(len(labels_new)):
#         a.write(labels_new[i] + '\n')