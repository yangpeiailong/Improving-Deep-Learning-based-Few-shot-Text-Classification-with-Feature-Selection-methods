from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from collections import OrderedDict
import math
from tqdm import tqdm
import pickle
from itertools import combinations
from nltk.tokenize import word_tokenize
import pandas as pd
import sys




def get_words(samples_temp):
    words_temp = []
    for line_temp in samples_temp:

        for word in line_temp:
            if str(word) not in words_temp:
                words_temp.append(word)
    return words_temp


def nCr(n,r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))

def dummy_fun(doc):
    return doc

def word_word_edges(p_ij):
    word_word = []
    cols = list(p_ij.columns)
    cols = [str(w) for w in cols]
    '''
    # old, inefficient but maybe more instructive code
    dum = []; counter = 0
    for w1 in tqdm(cols, total=len(cols)):
        for w2 in cols:
            #if (counter % 300000) == 0:
            #    print("Current Count: %d; %s %s" % (counter, w1, w2))
            if (w1 != w2) and ((w1,w2) not in dum) and (p_ij.loc[w1,w2] > 0):
                word_word.append((w1,w2,{"weight":p_ij.loc[w1,w2]})); dum.append((w2,w1))
            counter += 1
    '''
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1,w2] > 0):
            word_word.append((w1,w2,{"weight":p_ij.loc[w1,w2]}))
    return word_word




if __name__ == '__main__':
    dataname, num_of_sample, fs_method, num_of_features = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # 加载数据集
    # dataname = '20newsgroup'
    # num_of_sample = str(1000)
    sample_data_file = 'e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname \
                       + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features + '.txt'
    label_data_file = 'e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname \
                      + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features + '_label.txt'
    # sample_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '.txt'
    # label_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_label.txt'
    # sample_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_small.txt'
    # label_data_file = 'e:/pythonwork/newclassification/dataset/' + dataname + '_label_small.txt'
    samples = list(open(sample_data_file, "r", encoding='utf-8').readlines())
    samples = [s.strip() for s in samples]
    labels = list(open(label_data_file, "r", encoding='utf-8').readlines())
    labels = [s.strip() for s in labels]
    # print(samples[0])

    #
    vectorizer = TfidfVectorizer()
    vectorizer.fit(samples)
    df_tfidf = vectorizer.transform(samples)
    df_tfidf = df_tfidf.toarray()


    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    # print(vocab)
    #
    # # print(len(df_tfidf))
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)
    word2idx = OrderedDict((word,index) for index, word in enumerate(vocab))
    n_i = OrderedDict((word, 0) for word in vocab)
    # print(n_i['5'])

    # print(word2idx)

    window = 10

    occurrences = np.zeros((len(vocab), len(vocab)), dtype=np.int32)
    # Find the co-occurrences:
    no_windows = 0
    samples = [word_tokenize(sample) for sample in samples]
    # 由于tfidf工具自带去停用词过程，为了避免其删除后的词仍然在samples中存在，给samples中不在vocab中的词进行过滤
    samples = [[word for word in sentence if word in vocab] for sentence in samples ]
    with open('./GCN/middle_data/' + dataname + '_' + \
                       num_of_sample + '_' + fs_method + '_' + num_of_features + '.txt', 'w', encoding='utf-8') as f:
        for i in range(len(samples)):
            f.write(" ".join(samples[i]) + '\n')
    with open('./GCN/middle_data/' + dataname + '_' + \
              num_of_sample + '_' + fs_method + '_' + num_of_features + '_label.txt', 'w', encoding='utf-8') as f:
        for i in range(len(labels)):
            f.write(labels[i] + '\n')


    for sentence in tqdm(samples, total=len(samples)):
        if len(sentence) <= window:
            no_windows += 1
            d = set(sentence)
            for w in d:
                n_i[w] += 1
            for w1, w2 in combinations(d, 2):
                i1 = word2idx[w1]
                i2 = word2idx[w2]

                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1
        else:
            for i in range(len(sentence) - window):
                no_windows += 1
                d = set(sentence[i:(i + window)])

                for w in d:
                    n_i[w] += 1
                for w1, w2 in combinations(d, 2):
                    i1 = word2idx[w1]
                    i2 = word2idx[w2]

                    occurrences[i1][i2] += 1
                    occurrences[i2][i1] += 1

    p_ij = pd.DataFrame(occurrences, index=vocab, columns=vocab) / no_windows
    p_i = pd.Series(n_i, index=n_i.keys()) / no_windows

    del occurrences
    del n_i
    for col in p_ij.columns:
        p_ij[col] = p_ij[col] / p_i[col]
    for row in p_ij.index:
        p_ij.loc[row, :] = p_ij.loc[row, :] / p_i[row]
    p_ij = p_ij + 1E-9
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
    G = nx.Graph()
    G.add_nodes_from(df_tfidf.index)  ## document nodes
    G.add_nodes_from(vocab)  ## word nodes
    document_word = [(doc, w, {"weight": df_tfidf.loc[doc, w]}) for doc in
                     tqdm(df_tfidf.index, total=len(df_tfidf.index)) \
                     for w in df_tfidf.columns]
    word_word = word_word_edges(p_ij)
    G.add_edges_from(document_word)
    G.add_edges_from(word_word)
    f = open('./GCN/graph/'+ dataname + '_' + \
                       num_of_sample + '_' + fs_method + '_' + num_of_features + 'text_graph.pkl', 'wb')
    pickle.dump(G, f)
