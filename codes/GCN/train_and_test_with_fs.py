from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from collections import OrderedDict
import math
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import time
import xlwt
from torch import optim
import torch
import sys
from itertools import combinations
from nltk.tokenize import word_tokenize
import pandas as pd
from argparse import ArgumentParser
import torch.nn as nn
from models import gcn
import pickle

print('GPU:', torch.cuda.is_available())
torch.manual_seed(123)






def k_fold_cross_validation(doc_terms_list_temp, labels_temp, kk):
    skf = StratifiedKFold(n_splits=kk)
    skf.get_n_splits(doc_terms_list_temp, labels_temp)
    train_index_list = []
    test_index_list = []
    for train_index, test_index in skf.split(doc_terms_list_temp, labels_temp):
        train_index_list.append(train_index.tolist())
        test_index_list.append(test_index.tolist())
    return train_index_list, test_index_list

def measures(preds, y, y_matrix, y_sum):
    # y_preds = torch.argmax(preds, 1)
    # y_preds = y_preds.cpu().numpy()
    preds = np.array(preds)
    y_preds = np.argmax(preds, 1)
    n_class = preds.shape[1]
    # y = y.cpu().numpy()
    acc = metrics.accuracy_score(y, y_preds)

    if 0 in y_sum:
        auc = 0
    else:
        # print(y_matirx.shape, preds.shape)
        auc = metrics.roc_auc_score(y_matrix, preds, average='weighted')
    # recall = metrics.recall_score(y, y_preds, average='weighted')
    mavgtotal = 1
    mae = 0
    n_num = 0
    for iii in range(n_class):
        kkk = len([jjj for jjj in range(len(y)) if y[jjj] == iii])
        if kkk != 0:
            n_num += 1
            mavgtotal = mavgtotal * len([jjj for jjj in range(len(y)) if y[jjj] == iii
                                         and y[jjj] == np.argmax(preds[jjj])]) / kkk
        for jjj in range(len(preds)):
            if iii == y[jjj]:
                mae += abs(1.0 - preds[jjj][iii]) / (n_class * len(preds))
            else:
                mae += abs(preds[jjj][iii]) / (n_class * len(preds))
    mavg = pow(mavgtotal, 1.0 / n_num)
    # return acc, auc, mavg, mae, recall
    return acc, auc, mavg, mae

def eval():
    with torch.no_grad():
        testpred = net(f)
        testpred = testpred[not_selected]
        if pred.dim() > 1:
            testpred = testpred
        else:
            testpred = testpred.unsqueeze(0)
        test_y = labels_not_selected
        # print(len(test_y))
        # print(testpred.shape)

        ymatrix = np.zeros(testpred.shape, dtype=int).tolist()
        # print(ymatrix)
        testpred = testpred.cpu().numpy().tolist()
        for iii in range(len(test_y)):
            ymatrix[iii][test_y[iii]] = 1
        # print(testpred, ymatrix)
        total_class = np.sum(ymatrix, axis=0).tolist()
        acc_temp, auc_temp, mavg_temp, mae_temp = measures(testpred, test_y, ymatrix, total_class)
        print('>>test:', acc_temp, auc_temp, mavg_temp, mae_temp)
        return acc_temp, auc_temp, mavg_temp, mae_temp






if __name__ == '__main__':
    epoches, dataname, num_of_sample, fs_method, num_of_features, op, para = int(sys.argv[1]), sys.argv[2], sys.argv[3], \
                                                                             sys.argv[4], sys.argv[5], sys.argv[
                                                                                 6], float(sys.argv[7])
    # 加载数据集
    # dataname = '20newsgroup'
    # num_of_sample = str(1000)
    sample_data_file = './GCN/middle_data/' + dataname + '_' + \
                       num_of_sample + '_' + fs_method + '_' + num_of_features + '.txt'
    label_data_file = './GCN/middle_data/' + dataname + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features\
                      + '_label.txt'
    samples = list(open(sample_data_file, "r", encoding='utf-8').readlines())
    # samples = list(open(sample_data_file, "r").readlines())
    samples = [s.strip() for s in samples]
    labels = list(open(label_data_file, "r", encoding='utf-8').readlines())
    # labels = list(open(label_data_file, "r").readlines())
    labels = [s.strip() for s in labels]
    samples = [sentence.split() for sentence in samples]

    label2idx = {label: i for i, label in enumerate(list(set(labels)))}
    labels = [label2idx[i] for i in labels]

    f = open('./GCN/graph/'+ dataname + '_' + \
                       num_of_sample + '_' + fs_method + '_' + num_of_features + 'text_graph.pkl', 'rb')
    G = pickle.load(f)

    A = nx.to_numpy_matrix(G, weight="weight")
    # print(A.shape)
    A = A + np.eye(G.number_of_nodes())
    degrees = []

    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1] ** (-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes())  # Features are just identity matrix
    A_hat = degrees @ A @ degrees



    k = 5
    train_list, test_list = k_fold_cross_validation(samples, labels, k)
    # print(train_list)
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('test', cell_overwrite_ok=True)
    sheet.write(0, 0, 'classifier')
    sheet.write(0, 1, 'avg_of_acc')
    sheet.write(0, 2, 'std_of_acc')
    sheet.write(0, 3, 'avg_of_auc')
    sheet.write(0, 4, 'std_of_auc')
    sheet.write(0, 5, 'avg_of_mavg')
    sheet.write(0, 6, 'std_of_mavg')
    sheet.write(0, 7, 'avg_of_mae')
    sheet.write(0, 8, 'std_of_mae')

    sheet.write(0, 9, 'avg_of_training_time')
    sheet.write(0, 10, 'std_of_training_time')
    sheet.write(0, 11, 'avg_of_test_time')
    sheet.write(0, 12, 'std_of_test_time')

    measures_total = np.zeros((k, 6))
    for i in range(k):
        f = X  # (n X n) X (n X n) x (n X n) X (n X n) input of net
        selected = train_list[i]
        f_selected = np.array([f[ii] for ii in selected])
        f_selected = torch.from_numpy(f_selected).float()
        labels_selected = [labels[ii] for ii in selected]
        not_selected = test_list[i]
        f_not_selected = np.array([f[ii] for ii in not_selected])
        f_not_selected = torch.from_numpy(f_not_selected).float()
        labels_not_selected = [labels[ii] for ii in not_selected]
        f = torch.from_numpy(f).float()
        hidden_size_1 = 330
        hidden_size_2 = 130
        num_classes = len(label2idx)

        # parser = ArgumentParser()
        # parser.add_argument("--hidden_size_1", type=int, default=330, help="Size of first GCN hidden weights")
        # parser.add_argument("--hidden_size_2", type=int, default=130, help="Size of second GCN hidden weights")
        # parser.add_argument("--num_classes", type=int, default=len(label2idx), help="Number of prediction classes")
        # parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
        # parser.add_argument("--num_epochs", type=int, default=epoches, help="No of epochs")
        # parser.add_argument("--lr", type=float, default=para, help="learning rate")
        # parser.add_argument("--model_no", type=int, default=0, help="Model ID")
        # args = parser.parse_args()

        net = gcn(X.shape[1], A_hat, hidden_size_1, hidden_size_2, num_classes).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = 0
        if op == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=para)
        elif op == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=para)
        else:
            print('Wrong optimizer')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000, 5000, 6000],
                                                   gamma=0.77)

        best_val_acc = 0
        best_val_auc = 0
        best_val_mavg = 0
        best_val_mae = 0
        bestepoch = 0
        trainingstart = time.time()
        trainingend = 0
        for e in range(epoches):
            net.train()
            # for ii, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            f = f.cuda()
            pred = net(f)
            # print(len(pred), len(labels_selected))
            loss = criterion(pred[selected], torch.tensor(labels_selected).long().cuda())
            loss.backward()
            optimizer.step()
            if e % 50 == 0:
                print('第{}折，共{}折，分类器为GCN，第{}期，共{}期，损失为{}'.format(i + 1, k, e, epoches, loss.item()))
                acc, auc, mavg, mae = eval()
                if acc > best_val_acc:
                    torch.save(net.state_dict(),
                               'GCN/model/model_params_best_{data}_{samplenum}_{fsmethod}_{fsnum}_GCN_{k_num}.pkl'.format(
                                   data=dataname, samplenum=num_of_sample, fsmethod=fs_method, fsnum=num_of_features, k_num=i))
                    # torch.save(model.state_dict(), 'model/model_params_best_{data}_{modelname}_{k_num}.pkl'.format(data=dataname,modelname=classifier_list[j], k_num=i))
                    best_val_acc = acc
                    best_val_auc = auc
                    best_val_mavg = mavg
                    best_val_mae = mae
                    # best_val_recall = recall
                    bestepoch = e
                    trainingend = time.time()
        print('for k = %s' % i, 'bestepoch:', bestepoch, 'bestacc:', best_val_acc)
        teststart = time.time()
        net.load_state_dict(torch.load(
            'GCN/model/model_params_best_{data}_{samplenum}_{fsmethod}_{fsnum}_GCN_{k_num}.pkl'.format(data=dataname,
                                                                                        samplenum=num_of_sample, fsmethod=fs_method, fsnum=num_of_features,
                                                                                        k_num=i)))
        # model.load_state_dict(torch.load('model/model_params_best_{data}_{modelname}_{k_num}.pkl'.format(data=dataname,modelname=classifier_list[j], k_num=i)))
        print('for model = GCN, k = {k_num}'.format(k_num=k), 'test best model:')
        measures_total[i][0], measures_total[i][1], measures_total[i][2], measures_total[i][3] = eval()
        testend = time.time()
        measures_total[i][4] = trainingend - trainingstart
        measures_total[i][5] = testend - teststart
        print('最优表现：', dataname, measures_total[i][0], measures_total[i][1], measures_total[i][2],
              measures_total[i][3], measures_total[i][4], measures_total[i][5])
        print('=========================================')
    measures_total_avg = np.mean(measures_total, axis=0)
    measures_total_std = np.std(measures_total, axis=0)
    sheet.write(1, 0, "GCN")
    sheet.write(1, 1, measures_total_avg[0])
    sheet.write(1, 2, measures_total_std[0])
    sheet.write(1, 3, measures_total_avg[1])
    sheet.write(1, 4, measures_total_std[1])
    sheet.write(1, 5, measures_total_avg[2])
    sheet.write(1, 6, measures_total_std[2])
    sheet.write(1, 7, measures_total_avg[3])
    sheet.write(1, 8, measures_total_std[3])
    sheet.write(1, 9, measures_total_avg[4])
    sheet.write(1, 10, measures_total_std[4])
    sheet.write(1, 11, measures_total_avg[5])
    sheet.write(1, 12, measures_total_std[5])
    book.save('e:/pythonwork/newclassification/results_with_feature_selection/' + dataname + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features + '_GCN_criteria.xls')