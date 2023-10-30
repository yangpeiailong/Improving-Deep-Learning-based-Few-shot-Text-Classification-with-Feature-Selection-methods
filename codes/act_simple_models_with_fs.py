import jieba
import re
import numpy as np
import torch.utils.data as tud
import torch
from torch import optim, nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import time
# from tqdm import tqdm
import xlwt
import fasttext.util
from fasttext import load_model

import sys
print('GPU:', torch.cuda.is_available())
torch.manual_seed(123)



# def get_words(samples_temp):
#     words_temp = []
#     for line_temp in samples_temp:
#         tmp_list = list(line_temp.split(' '))
#         for word in tmp_list:
#             if str(word) not in words_temp:
#                 words_temp.append(word)
#     return words_temp

def get_words(samples_temp):
    words_temp = []
    for line_temp in samples_temp:

        for word in line_temp:
            if str(word) not in words_temp:
                words_temp.append(word)
    return words_temp



def k_fold_cross_validation(doc_terms_list_temp, labels_temp, kk):
    skf = StratifiedKFold(n_splits=kk)
    skf.get_n_splits(doc_terms_list_temp, labels_temp)
    doc_train_list_temp = []
    labels_train_list_temp = []
    doc_test_list_temp = []
    labels_test_list_temp = []
    for train_index, test_index in skf.split(doc_terms_list_temp, labels_temp):
        # print("TRAIN:",train_index, "TEST:", test_index)
        x_train, x_test = np.array(doc_terms_list_temp, dtype=object)[train_index], np.array(doc_terms_list_temp, dtype=object)[test_index]
        y_train, y_test = np.array(labels_temp)[train_index], np.array(labels_temp)[test_index]
        doc_train_list_temp.append(x_train.tolist())
        labels_train_list_temp.append(y_train.tolist())
        doc_test_list_temp.append(x_test.tolist())
        labels_test_list_temp.append(y_test.tolist())
    return doc_train_list_temp, labels_train_list_temp, doc_test_list_temp, labels_test_list_temp



# samples_encoder = torch.tensor(samples_encoder)
# labels = torch.tensor(labels)


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self, x_data, y_data):
        super(DealDataset, self).__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def measures(preds, y, y_matirx, y_sum_temp):
    # y_preds = torch.argmax(preds, 1)
    # y_preds = y_preds.cpu().numpy()
    preds = np.array(preds)
    y_preds = np.argmax(preds, 1)
    n_class = preds.shape[1]
    # y = y.cpu().numpy()
    acc = metrics.accuracy_score(y, y_preds)

    if 0 in y_sum_temp:
        auc = 0
    else:
        auc = metrics.roc_auc_score(y_matirx, preds, average='weighted')
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


def pad(x):
    return x[:MAX_LENGHT] if len(x) > MAX_LENGHT else x + [0] * (MAX_LENGHT - len(x))


def train():
    sum_of_loss = 0
    model.train()
    for ii, data in enumerate(dataloader_train):
        optimizer.zero_grad()
        x, y = data
        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        if pred.dim() > 1:
            pred = pred
        else:
            pred = pred.unsqueeze(0)
        # print(pred.shape, y.shape)
        loss = criteon(pred, y)
        loss.backward()
        optimizer.step()
        sum_of_loss += loss.item()
        # print(loss.item())

    return sum_of_loss

def eval():
    model.eval()
    pred_list = []
    y_list = []
    y_temp_list = []
    # acc_list = []
    # auc_list = []
    # mavg_list = []
    # mae_list = []
    # recall_list = []
    with torch.no_grad():
        for ii, data in enumerate(dataloader_eval):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            # [b, 1] => [b]
            pred = model(x)
            if pred.dim() > 1:
                pred = pred
            else:
                pred = pred.unsqueeze(0)
            y = y.cpu().numpy().tolist()
            y_class = []
            for iii in range(len(y)):
                if y[iii] not in y_class:
                    y_class.append(y[iii])
            y_temp = np.zeros(pred.shape).tolist()
            pred = pred.cpu().numpy().tolist()
            for iii in range(len(y)):
                y_temp[iii][y[iii]] = 1
            # print(y_temp)
            # y_sum = np.sum(y_temp, axis=0).tolist()
            # if 0 in y_sum:
            #     print('batch中的样本没有覆盖全部类，无法计算auc，忽略该batch的计算')
            #     continue
            #
            # loss = criteon(pred, y.float())
            # acc = binary_acc(pred, y.float()).item()
            # acc, auc, mavg, mae, recall = measures(pred, y)
            # acc, auc, mavg, mae = measures(pred, y, y_sum)
            # acc_list.append(acc)
            # auc_list.append(auc)
            # mavg_list.append(mavg)
            # mae_list.append(mae)
            # recall_list.append(recall)
            pred_list += pred
            y_list += y
            y_temp_list += y_temp

        # avg_acc = np.array(acc_list).mean()
        # avg_auc = np.array(auc_list).mean()
        # avg_mavg = np.array(mavg_list).mean()
        # avg_mae = np.array(mae_list).mean()
        y_sum = np.sum(y_temp_list, axis=0).tolist()
        print(y_sum)
        acc_temp, auc_temp, mavg_temp, mae_temp = measures(pred_list, y_list, y_temp_list, y_sum)
        # avg_recall = np.array(recall_list).mean()
        # print('>>test:', avg_acc, avg_auc, avg_mavg, avg_mae, avg_recall)
        # return avg_acc, avg_auc, avg_mavg, avg_mae, avg_recall
        # print('>>test:', avg_acc, avg_auc, avg_mavg, avg_mae)
        print('>>test:', acc_temp, auc_temp, mavg_temp, mae_temp)
        return acc_temp, auc_temp, mavg_temp, mae_temp


if __name__ == '__main__':
    # 加载数据集
    epoches, dataname, num_of_sample, fs_method, num_of_features, \
    op1, para1, op2, para2, op3, para3, op4, para4, op5, para5, op6, para6, \
    op7, para7, op8, para8, op9, para9, op10, para10 \
        = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], \
          sys.argv[6], float(sys.argv[7]), sys.argv[8], float(sys.argv[9]), sys.argv[10], float(sys.argv[11]), \
          sys.argv[12], float(sys.argv[13]), sys.argv[14], float(sys.argv[15]), sys.argv[16], float(sys.argv[17]), \
          sys.argv[18], float(sys.argv[19]), sys.argv[20], float(sys.argv[21]), sys.argv[22], float(sys.argv[23]), \
          sys.argv[24], float(sys.argv[25])
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
    from nltk.tokenize import word_tokenize
    samples = [word_tokenize(sample) for sample in samples]
    # 获取所有词
    words = get_words(samples)
    # print(len(words))
    #
    # MAX_VOCAB_SIZE = 30000
    embed_size = 100
    batch_size_training = 32
    batch_size_test = 100

    wordsdf_dict = {}
    for i in range(len(words)):
        wordsdf_dict[words[i]] = 0

    for i in range(len(words)):
        for j in range(len(samples)):
            if words[i] in samples[j]:
                wordsdf_dict[words[i]] += 1

    # print(wordsdf_dict)
    wordsdf = sorted(wordsdf_dict.items(), key=lambda d:d[1], reverse=True)

    words2 = []
    for i in range(len(wordsdf)):
        words2.append(wordsdf[i][0])


    samples_split = samples
    # for i in range(len(samples)):
    #     tmp_list = list(samples[i].split(' '))
    #     for j in range(len(tmp_list)):
    #         if tmp_list[j] not in words2:
    #             tmp_list[j] = '<unk>'
    #     samples_split.append(tmp_list)
    # print(samples_split[0], len(samples_split[0]))
    # print(len(samples), len(labels))


    # 读取fasttext预训练的词向量，注意路径中不要包含中文，否则会报错
    print('开始加载词向量！')
    loadvecstart = time.time()
    ft = load_model('E:/fasttext/cc.en.300.bin')
    # 将词向量维度减少到embed_size维
    fasttext.util.reduce_model(ft, embed_size)
    ftwordvec = []

    for i in words2:
        ftwordvec.append(ft.get_word_vector(i))
    # 增加<pad>和<unk>的词向量，都是初始化为所有维度都为0
    ftwordvec.append([0.0] * embed_size)
    ftwordvec.append([0.0] * embed_size)
    loadvecend = time.time()
    print('加载词向量的时间：', loadvecend - loadvecstart, '秒')
    # ftwordvec 存储来自于fasttext的词向量（按照words2中间词的顺序排列）

    pretrained_weight = np.array(ftwordvec)
    # print(words2[0], ftwordvec[0])
    labelcase = []
    for i in labels:
        if i not in labelcase:
            labelcase.append(i)
    numofclasses = len(labelcase)
    label2ind = {label: i for i, label in enumerate(labelcase)}
    with open(
            'model/label_to_index_dl_for_' + dataname + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features + '.txt',
            'w', encoding='utf-8') as f:
        for i in label2ind.keys():
            f.writelines(str(i) + ':' + str(label2ind[i]) + '\n')
        f.close()
    # with open('model/label_to_index_dl_for_' + dataname + '.txt', 'w', encoding='utf-8') as f:
    #     for i in label2ind.keys():
    #         f.writelines(str(i) + ':' + str(label2ind[i]) + '\n')
    #     f.close()
    labels = [label2ind[label] for label in labels]
    # word_to_idx
    word_to_idx = {word: i for i, word in enumerate(words2)}
    word_to_idx['<pad>'] = len(words2)
    word_to_idx['<unk>'] = len(words2) + 1
    with open('model/word_to_index_dl_for_'+ dataname + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features +'.txt', 'w', encoding='utf-8') as f:
        for i in word_to_idx.keys():
            f.writelines(i + ':' + str(word_to_idx[i]) + '\n')
        f.close()

    # with open('model/word_to_index_dl_for_'+ dataname + '.txt', 'w', encoding='utf-8') as f:
    #     for i in word_to_idx.keys():
    #         f.writelines(i + ':' + str(word_to_idx[i]) + '\n')
    #     f.close()
    pad_index = word_to_idx['<pad>']
    # 将原始文本转化为词索引文本
    samples_encoder = []
    for i in range(len(samples_split)):
        enctemp = []
        for j in range(len(samples_split[i])):
            enctemp.append(word_to_idx[samples_split[i][j]])
        samples_encoder.append(enctemp)
    # print(len(samples_encoder[0]))

    MAX_LENGHT = min([max([len(sample) for sample in samples_encoder]), 200])


    # 给长度不足MAX_LENGHTH的文本补0元素的函数
    for i in range(len(samples_encoder)):
        samples_encoder[i] = pad(samples_encoder[i])
    k = 5  # 确定 k-fold cross validation的k
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
    samples_train_list, labels_train_list, samples_eval_list, labels_eval_list = k_fold_cross_validation(samples_encoder, labels, k)
    numoffilters = 100
    filtersizes = [2, 3, 4]
    hiddensize = 100
    epoches = 100
    classifier_list = ['random+WA', 'FastText+WA', 'random+CNN', 'FastText+CNN', 'random+LSTM', 'FastText+LSTM', 'random+LSTM+att', 'FastText+LSTM+att', 'random+LSTM+CNN', 'FastText+LSTM+CNN']
    # classifier_list = ['random+LSTM']
    # measures_total = np.zeros((k, len(classifier_list), 7))
    measures_total = np.zeros((k, len(classifier_list), 6))
    # measures_total_avg = np.zeros((len(classifier_list), 7))
    # measures_total_std = np.zeros((len(classifier_list), 7))

    for i in range(k):
        # print('第%s折' % i)
        samples_train = torch.tensor(samples_train_list[i])
        labels_train = torch.tensor(labels_train_list[i])
        samples_eval = torch.tensor(samples_eval_list[i])
        labels_eval = torch.tensor(labels_eval_list[i])
        dataset_train = DealDataset(samples_train, labels_train)
        dataset_eval = DealDataset(samples_eval, labels_eval)
        dataloader_train = tud.DataLoader(dataset_train, batch_size=batch_size_training, shuffle=True, num_workers=0)
        dataloader_eval = tud.DataLoader(dataset_eval, batch_size=batch_size_test, shuffle=True, num_workers=0)
        for j in range(len(classifier_list)):

            model = 0
            optimizer = 0
            criteon = 0
            device = torch.device('cuda')
            if classifier_list[j] == 'random+WA':
                from DeepLearningTC.simple_models import WordAveraging
                model = WordAveraging(len(word_to_idx), embed_size, numofclasses, pad_index)
                if op1 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para1)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para1)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'FastText+WA':
                from DeepLearningTC.simple_models import WordAveraging
                model = WordAveraging(len(word_to_idx), embed_size, numofclasses, pad_index, pretrained_weight)
                if op2 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para2)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para2)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'random+CNN':
                from DeepLearningTC.simple_models import TextCNN
                model = TextCNN(len(word_to_idx), embed_size, numoffilters, filtersizes, MAX_LENGHT, numofclasses,
                                pad_index)
                if op3 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para3)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para3)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'FastText+CNN':
                from DeepLearningTC.simple_models import TextCNN
                model = TextCNN(len(word_to_idx), embed_size, numoffilters, filtersizes, MAX_LENGHT, numofclasses, pad_index,
                                pretrained_weight)
                if op4 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para4)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para4)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'random+LSTM':
                from DeepLearningTC.simple_models import TextLSTM
                model = TextLSTM(len(word_to_idx), embed_size, hiddensize, numofclasses, pad_index)
                if op5 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para5)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para5)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'FastText+LSTM':
                from DeepLearningTC.simple_models import TextLSTM
                model = TextLSTM(len(word_to_idx), embed_size, hiddensize, numofclasses, pad_index, pretrained_weight)
                if op6 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para6)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para6)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'random+LSTM+att':
                from DeepLearningTC.simple_models import TextLSTMAttention
                model = TextLSTMAttention(len(word_to_idx), embed_size, hiddensize, numofclasses, pad_index)
                if op7 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para7)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para7)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'FastText+LSTM+att':
                from DeepLearningTC.simple_models import TextLSTMAttention
                model = TextLSTMAttention(len(word_to_idx), embed_size, hiddensize, numofclasses, pad_index, pretrained_weight)
                if op8 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para8)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para8)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'random+LSTM+CNN':
                from DeepLearningTC.simple_models import TextLSTMCNN
                model = TextLSTMCNN(len(word_to_idx), embed_size, numoffilters, filtersizes, MAX_LENGHT, hiddensize, numofclasses,
                                pad_index)
                if op9 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para9)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para9)
                criteon = nn.CrossEntropyLoss().cuda()
            elif classifier_list[j] == 'FastText+LSTM+CNN':
                from DeepLearningTC.simple_models import TextLSTMCNN
                model = TextLSTMCNN(len(word_to_idx), embed_size, numoffilters, filtersizes, MAX_LENGHT, hiddensize, numofclasses, pad_index,
                                pretrained_weight)
                if op10 == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=para10)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=para10)
                criteon = nn.CrossEntropyLoss().cuda()
            # optimizer = optim.Adam(model.parameters(), lr=1e-3)
            # criteon = nn.CrossEntropyLoss().cuda()
            model = model.cuda()
            best_val_acc = 0
            best_val_auc = 0
            best_val_mavg = 0
            best_val_mae = 0
            # best_val_recall = 0
            bestepoch = 0
            trainingstart = time.time()
            trainingend = 0
            for epoch in range(epoches):
                sumloss = train()
                # print(classifier_list[j])
                print('第{}折，共{}折，分类器为{}，第{}期，共{}期，损失为{}'.format(i + 1, k, classifier_list[j], epoch, epoches, sumloss))
                # 挑选最优参数
                # acc, auc, mavg, mae, recall = eval()
                acc, auc, mavg, mae = eval()
                if acc > best_val_acc:
                    torch.save(model.state_dict(),
                               'model/model_params_best_{data}_{samplenum}_{fsmethod}_{fsnum}_{modelname}_{k_num}.pkl'.format(
                                   data=dataname, samplenum=num_of_sample, fsmethod=fs_method, fsnum=num_of_features,
                                   modelname=classifier_list[j], k_num=i))
                    # torch.save(model.state_dict(), 'model/model_params_best_{data}_{modelname}_{k_num}.pkl'.format(data=dataname,modelname=classifier_list[j], k_num=i))
                    best_val_acc = acc
                    best_val_auc = auc
                    best_val_mavg = mavg
                    best_val_mae = mae
                    # best_val_recall = recall
                    bestepoch = epoch
                    trainingend = time.time()
            print('for k = %s'% i, 'bestepoch:', bestepoch, 'bestacc:', best_val_acc)
            # 载入并测试
            teststart = time.time()
            model.load_state_dict(torch.load(
                'model/model_params_best_{data}_{samplenum}_{fsmethod}_{fsnum}_{modelname}_{k_num}.pkl'.format(
                    data=dataname, samplenum=num_of_sample, fsmethod=fs_method, fsnum=num_of_features,
                    modelname=classifier_list[j], k_num=i)))
            # model.load_state_dict(torch.load('model/model_params_best_{data}_{modelname}_{k_num}.pkl'.format(data=dataname,modelname=classifier_list[j], k_num=i)))
            print('for model = {modelname}, k = {k_num}'.format(modelname=classifier_list[j], k_num=k), 'test best model:')
            measures_total[i][j][0], measures_total[i][j][1], measures_total[i][j][2], measures_total[i][j][3] = eval()
            testend = time.time()
            measures_total[i][j][4] = trainingend - trainingstart
            measures_total[i][j][5] = testend - teststart
            print('最优表现：', measures_total[i][j][0], measures_total[i][j][1], measures_total[i][j][2],
                  measures_total[i][j][3], measures_total[i][j][4], measures_total[i][j][5])
            print('=========================================')
    measures_total_avg = np.mean(measures_total, axis=0)
    measures_total_std = np.std(measures_total, axis=0)
    for i in range(len(measures_total_avg)):
        sheet.write(i + 1, 0, classifier_list[i])
        sheet.write(i + 1, 1, measures_total_avg[i][0])
        sheet.write(i + 1, 2, measures_total_std[i][0])
        sheet.write(i + 1, 3, measures_total_avg[i][1])
        sheet.write(i + 1, 4, measures_total_std[i][1])
        sheet.write(i + 1, 5, measures_total_avg[i][2])
        sheet.write(i + 1, 6, measures_total_std[i][2])
        sheet.write(i + 1, 7, measures_total_avg[i][3])
        sheet.write(i + 1, 8, measures_total_std[i][3])
        # sheet.write(i + 1, 9, measures_total_avg[i][4])
        # sheet.write(i + 1, 10, measures_total_std[i][4])
        # sheet.write(i + 1, 11, measures_total_avg[i][5])
        # sheet.write(i + 1, 12, measures_total_std[i][5])
        # sheet.write(i + 1, 13, measures_total_avg[i][6])
        # sheet.write(i + 1, 14, measures_total_std[i][6])
        sheet.write(i + 1, 9, measures_total_avg[i][4])
        sheet.write(i + 1, 10, measures_total_std[i][4])
        sheet.write(i + 1, 11, measures_total_avg[i][5])
        sheet.write(i + 1, 12, measures_total_std[i][5])

    book.save(
        'e:/pythonwork/newclassification/results_with_feature_selection/' + dataname + '_' + num_of_sample
        + '_' + fs_method + '_' + num_of_features + '_dl_criteria.xls')
    # book.save('e:/pythonwork/newclassification/results/' + dataname + '_dl_criteria.xls')