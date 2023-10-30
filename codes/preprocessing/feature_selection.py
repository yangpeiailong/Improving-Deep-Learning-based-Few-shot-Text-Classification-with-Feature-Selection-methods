from tqdm import tqdm
import numpy as np
import math

dataname = 'WOS5736'
num_of_sample = str(1000)
feature_numbers = 500
fs_method = 'DFS'
with open('e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_' + num_of_sample +'.txt', 'r', encoding='utf-8') as a:
    texts = [i.strip().lower() for i in a.readlines()]
with open('e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_' + num_of_sample +'_label.txt', 'r', encoding='utf-8') as a:
    labels = [i.strip() for i in a.readlines()]

# with open('e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '.txt', 'r', encoding='utf-8') as a:
#     texts = [i.strip().lower() for i in a.readlines()]
# with open('e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname +'_label.txt', 'r', encoding='utf-8') as a:
#     labels = [i.strip() for i in a.readlines()]



from nltk.tokenize import word_tokenize
texts = [word_tokenize(sentence) for sentence in texts]

# 获得词汇表
def get_vocab(text):
    voc = []
    for i in text:
        for j in i:
            if j not in voc:
                voc.append(j)
    return voc


vocab = get_vocab(texts)
# print(len(vocab))
labels_class = list(set(labels))
# print(labels_class)
labels_number = {i: 0 for i in labels_class}
for i in labels:  # 1、-1
    labels_number[i] += 1
# print(labels_number)
N = len(texts)


def df():
    DF = {i: 0 for i in vocab}
    for i in range(len(vocab)):  # 14000+
        for j in range(len(texts)):  # 10600+
            if vocab[i] in texts[j]:
                DF[vocab[i]] += 1
    return DF


def ig():
    IG = {}
    for i in tqdm(range(len(vocab))):
        n_tc = np.zeros((2, len(labels_class)))  # [[0, 0], [0, 0]]
        for j in range(len(texts)):
            if vocab[i] in texts[j]:
                n_tc[0, labels_class.index(labels[j])] += 1
            else:
                n_tc[1, labels_class.index(labels[j])] += 1
        # print(n_tc)
        n_t = sum(n_tc[0, :])
        n_minorst = sum(n_tc[1, :])
        a = -sum([labels_number[i] / N * math.log(labels_number[i] / N + 1e-5) for i in labels_class])
        b = n_t / N * sum([n_tc[0, iii] / n_t * math.log(n_tc[0, iii] / n_t + 1e-5) for iii in
                           range(len(labels_class))]) if n_t != 0 else 0
        c = n_minorst / N * sum([n_tc[1, iii] / n_minorst * math.log(n_tc[1, iii] / n_minorst + 1e-5) for iii in
                                 range(len(labels_class))]) if n_minorst != 0 else 0
        ig = a + b + c
        IG[vocab[i]] = ig
    return IG


def dfs():
    DFS = {}
    for i in tqdm(range(len(vocab))):
        n_tc = np.zeros((2, len(labels_class)))  # [[0, 0], [0, 0]]
        for j in range(len(texts)):
            if vocab[i] in texts[j]:
                n_tc[0, labels_class.index(labels[j])] += 1
            else:
                n_tc[1, labels_class.index(labels[j])] += 1
        n_t = sum(n_tc[0, :])
        dfs = sum([n_tc[0, iii]/n_t/   # 分子
                   (n_tc[1, iii]/labels_number[labels_class[iii]] +  # 分母第一项
                    (n_t - n_tc[0, iii])/((N - labels_number[labels_class[iii]])/N) +  # 分母第二项
                    1) for iii in range(len(labels_class))])
        DFS[vocab[i]] = dfs
    return DFS


def get_features():
    feature_weights = {}
    if fs_method == 'IG':
        feature_weights = ig()
    elif fs_method == 'DF':
        feature_weights = df()
    elif fs_method == 'DFS':
        feature_weights = dfs()
    else:
        print('wrong name of method')
    features_after_sorting = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
    features = [word[0] for word in features_after_sorting[:feature_numbers]]
    return features


features = get_features()

# with open('D:/成都理工大学重要文件夹/课程《文本分析与挖掘》相关/数据/Pang&Lee_features.txt', 'w') as a:
#     for i in features:
#         a.write(i + '\n')
text_after_feature_selection = []
for i in texts:
    sentence = []
    for j in i:
        if j in features:
            sentence.append(j)
    text_after_feature_selection.append(sentence)

text_after_feature_selection2 = []
labels_after_feature_selection = []
for i in range(len(text_after_feature_selection)):
    if len(text_after_feature_selection[i]) > 0:
        text_after_feature_selection2.append(text_after_feature_selection[i])
        labels_after_feature_selection.append(labels[i])

feature_numbers = str(feature_numbers)

with open('e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname + '_' + num_of_sample + '_' + fs_method + '_' + feature_numbers +'.txt', 'w', encoding='utf-8') as a:
    for i in range(len(text_after_feature_selection2)):
        a.write(' '.join(text_after_feature_selection2[i]) + '\n')
with open('e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname + '_' + num_of_sample + '_' + fs_method + '_' + feature_numbers +'_label.txt', 'w', encoding='utf-8') as a:
    for i in range(len(labels_after_feature_selection)):
        a.write(' '.join(labels_after_feature_selection[i]) + '\n')

# with open('e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname + '_' + 'IG' + '_' + feature_numbers + '.txt', 'w', encoding='utf-8') as a:
#     for i in range(len(text_after_feature_selection2)):
#         a.write(' '.join(text_after_feature_selection2[i]) + '\n')
# with open('e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname + '_' + 'IG' + '_' + feature_numbers  +'_label.txt', 'w', encoding='utf-8') as a:
#     for i in range(len(labels_after_feature_selection)):
#         a.write(' '.join(labels_after_feature_selection[i]) + '\n')

