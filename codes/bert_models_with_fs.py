# -*- coding:utf-8 -*-
# bert文本分类模型
# model: bert
# date: 2022.3.29 10:36

import numpy as np
import jieba
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import StratifiedKFold
import xlwt
from sklearn import metrics
import time
import re
import torch.nn.functional as F
import math


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


def get_words(samples_temp):
    words_temp = []
    for line_temp in samples_temp:
        tmp_list = list(line_temp.split(' '))
        for word in tmp_list:
            if str(word) not in words_temp:
                words_temp.append(word)
    return words_temp


class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True, ):
        self.sentences = sentences
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = self.sentences[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        if self.with_labels:  # True if the dataset has labels
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

# model1
class BaseBert(nn.Module):
    def __init__(self):
        super(BaseBert, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_size, n_class)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        logits = self.softmax(self.linear(self.dropout(outputs.pooler_output)))
        # logits = self.linear(self.dropout(outputs.pooler_output))
        return logits


# model2
class BertWA(nn.Module):
    def __init__(self):
        super(BertWA, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.linear = nn.Linear(hidden_size, n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        outputs = outputs.last_hidden_state
        # x = x.detach()
        outputs = torch.mean(outputs, dim=1)
        outputs = self.linear(outputs)
        return self.softmax(outputs)

# model3
class BertCNN(nn.Module):
    def __init__(self):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=hidden_size, out_channels=num_of_filters, kernel_size=h),
                           nn.ReLU(),
                           nn.MaxPool1d(maxlen - h + 1)
                           ) for h in window_sizes]
        )
        self.linear = nn.Linear(num_of_filters * len(window_sizes), n_class)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        outputs = self.dropout(outputs.last_hidden_state)
        outputs = outputs.permute(0, 2, 1)
        outputs = [conv(outputs) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(-1, outputs.size(1))
        outputs = self.linear(outputs)
        return self.softmax(outputs)

# model4
class BertLSTM(nn.Module):
    def __init__(self):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.lstm = nn.LSTM(hidden_size, hidden_size_lstm, bidirectional=True)

        self.f1 = nn.Sequential(nn.Linear(hidden_size_lstm * 2, 128),
                                nn.Dropout(0.8),
                                nn.ReLU(),
                                nn.Linear(128, n_class),
                                nn.Softmax(dim=-1)
                                )


    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        x = self.dropout(outputs.last_hidden_state)
        x = x.permute(1, 0, 2)
        h0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        c0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # output, (hn, cn) = self.lstm(x)
        # print(hn.shape())
        # hn = torch.cat((hn[0], hn[1]), 1)
        # x = self.f1(hn)
        x = self.f1(output[-1])
        return x


# model5
class BertLSTMAttention(nn.Module):
    def __init__(self):
        super(BertLSTMAttention, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.lstm = nn.LSTM(hidden_size, hidden_size_lstm, bidirectional=True)

        # 定义计算注意力权重的内容
        self.linear1 = nn.Linear(hidden_size_lstm * 2, 128)
        self.tanh = nn.Tanh()
        self.u_w = nn.Linear(128, 1)
        self.softmax1 = nn.Softmax(dim=-1)

        # 定义输出
        self.f1 = nn.Sequential(nn.Linear(hidden_size_lstm * 2, 128),
                                nn.Dropout(0.8),
                                nn.ReLU(),
                                nn.Linear(128, n_class),
                                nn.Softmax(dim=-1)
                                )

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        x = self.dropout(outputs.last_hidden_state)
        x = x.permute(1, 0, 2)
        h0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        c0 = torch.randn(2, x.size(1), hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output.permute(1, 0, 2)
        # (seq_len, batchsize, hidden_size*num_directions) => (batchsize, seq_len, hidden_size*num_directions)

        attention_u = self.tanh(self.linear1(output))
        # (batchsize, seq_len, hidden_size*num_directions) => (batchsize, seq_len, u_size)
        attention_a = self.softmax1(self.u_w(attention_u))
        # print(attention_a.shape, output.shape)
        # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
        output = torch.matmul(attention_a.permute(0, 2, 1), output).squeeze()
        # (batchsize, 1, seq_len) * (batchsize, seq_len, hidden_size * num_directions) =>(batchsize, hidden_size * num_directions)
        # Q = self.Q(output)
        # K = self.K(output)
        # V = self.V(output)
        # att = torch.matmul(F.softmax(torch.matmul(Q.permute(0, 2, 1), K) / math.sqrt(Q.size(-1)), dim=-1), V).sum(1)

        return self.f1(output)
        # return self.f1(att)


# model6
class BertLSTMCNN(nn.Module):
    def __init__(self):
        super(BertLSTMCNN, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(hidden_size, n_class)
        self.lstm = nn.LSTM(hidden_size, hidden_size_lstm, bidirectional=True)

        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=hidden_size + 2 * hidden_size_lstm, out_channels=num_of_filters, kernel_size=h),
                           nn.ReLU(),
                           nn.MaxPool1d(maxlen - h + 1)
                           ) for h in window_sizes]
        )
        self.linear = nn.Linear(num_of_filters * len(window_sizes), n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [batch_size, hidden_size]
        x = self.dropout(outputs.last_hidden_state)
        input = x.permute(1, 0, 2)
        h0 = torch.randn(2, input.size(1), hidden_size_lstm).cuda()
        c0 = torch.randn(2, input.size(1), hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(input, (h0, c0))
        output = output.permute(1, 0, 2)
        # print(output.shape)
        outputsplit1, outputsplit2 = output.chunk(2, dim=2)
        # print(x.shape)
        outputcat = torch.cat((outputsplit1, x, outputsplit2), dim=2)
        outputcat = outputcat.permute(0, 2, 1)
        x = [conv(outputcat) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        return self.softmax(x)


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


def train():
    bc.train()
    sum_loss = 0
    for m, batch in enumerate(dataset_train):
        optimizer.zero_grad()
        batch = tuple(p.to(device) for p in batch)
        # print(batch[0].shape)
        pred = bc([batch[0], batch[1], batch[2]])
        # print(pred)
        if len(pred.shape) < 2:
            pred = pred.unsqueeze(0)
        # print(pred.shape, batch[3].shape)
        loss = loss_fn(pred, batch[3])
        # print(loss.item())
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_curve.append(sum_loss)
    # sum_loss = 0
    return sum_loss


def eval():
    bc.eval()
    pred_list = []
    y_list = []
    y_temp_list = []
    # acc_list = []
    # auc_list = []
    # mavg_list = []
    # mae_list = []
    # recall_list = []
    with torch.no_grad():
        for m, batch in enumerate(dataset_eval):
            batch = tuple(p.to(device) for p in batch)
            pred = bc([batch[0], batch[1], batch[2]])
            # print(batch[0], batch[1], batch[2], batch[3])
            # y = batch[3].cpu().numpy()
            y = batch[3].cpu().numpy().tolist()

            y_class = []
            for iii in range(len(y)):
                if y[iii] not in y_class:
                    y_class.append(y[iii])
            y_temp = np.zeros(pred.shape).tolist()
            pred = pred.cpu().numpy().tolist()
            for ii in range(len(y)):
                y_temp[ii][y[ii]] = 1
            # print(y_temp)
            # y_sum = np.sum(y_temp, axis=0).tolist()
            pred_list += pred
            y_list += y
            y_temp_list += y_temp

            # acc_temp, auc_temp, mavg_temp, mae_temp, recall_temp = measures(pred, y, y_temp)
            # acc_temp, auc_temp, mavg_temp, mae_temp = measures(pred, y, y_temp, y_sum)
            # acc_list.append(acc_temp)
            # auc_list.append(auc_temp)
            # mavg_list.append(mavg_temp)
            # mae_list.append(mae_temp)
            # recall_list.append(recall_temp)
    y_sum = np.sum(y_temp_list, axis=0).tolist()
    print(y_sum)
    # avg_acc = np.array(acc_list).mean()
    # avg_auc = np.array(auc_list).mean()
    # avg_mavg = np.array(mavg_list).mean()
    # avg_mae = np.array(mae_list).mean()
    acc_temp, auc_temp, mavg_temp, mae_temp = measures(pred_list, y_list, y_temp_list, y_sum)
    # avg_recall = np.array(recall_list).mean()
    # print('>>test:', avg_acc, avg_auc, avg_mavg, avg_mae, avg_recall)
    # return avg_acc, avg_auc, avg_mavg, avg_mae, avg_recall
    # print('>>test:', avg_acc, avg_auc, avg_mavg, avg_mae)
    print('>>test:', acc_temp, auc_temp, mavg_temp, mae_temp)
    return acc_temp, auc_temp, mavg_temp, mae_temp


if __name__ == '__main__':
    train_curve = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size_training = 32
    batch_size_test = 100
    epoches = 40
    model = "E:/bert-base-uncased"
    hidden_size = 768


    # data
    # sentences = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情"]
    # labels = [1, 1, 1, 0, 0, 0]  # 1积极, 0消极.

    # 加载数据集

    dataname = '20newsgroup'
    num_of_sample = str(1000)
    fs_method = 'IG'
    num_of_features = str(2000)
    # sample_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_' + num_of_sample + '.txt'
    # label_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname + '_' + num_of_sample + '_label.txt'

    sample_data_file = 'e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname \
                       + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features + '.txt'
    label_data_file = 'e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname \
                      + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features + '_label.txt'
    # sample_data_file = 'e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname \
    # + '_' + fs_method + '_' + num_of_features + '.txt'
    # label_data_file = 'e:/pythonwork/newclassification/dataset_after_feature_selection/' + dataname \
    # + '_' + fs_method + '_' + num_of_features + '_label.txt'
    # sample_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname  + '.txt'
    # label_data_file = 'e:/pythonwork/newclassification/dataset_after_preprocessing/' + dataname  + '_label.txt'
    # sample_data_file = 'E:/数据/' + dataname + '.txt'
    # label_data_file = 'E:/数据/' + dataname + '_label.txt'
    samples = list(open(sample_data_file, "r", encoding='utf-8').readlines())
    samples = [s.strip() for s in samples]
    labels = list(open(label_data_file, "r", encoding='utf-8').readlines())
    labels = [s.strip() for s in labels]
    from nltk.tokenize import word_tokenize
    samples = [word_tokenize(sample) for sample in samples]

    # print(samples[0])
    # maxlen = min([max([len(sample) for sample in samples]), 200])
    maxlen = max([len(sample) for sample in samples])
    print('maxlen:', maxlen)
    maxlen = min(maxlen, 200)
    print('maxlen:', maxlen)
    samples = [' '.join(sample) for sample in samples]
    # maxlen = 50
    labelcase = []
    for i in labels:
        if i not in labelcase:
            labelcase.append(i)
    n_class = len(labelcase)
    label2ind = {label: i for i, label in enumerate(labelcase)}
    with open(
            'model/label_to_index_bert_for_' + dataname + '_' + num_of_sample + '_' + fs_method + '_' + num_of_features + '.txt',
            'w', encoding='utf-8') as f:
        for i in label2ind.keys():
            f.writelines(str(i) + ':' + str(label2ind[i]) + '\n')
        f.close()
    # with open('model/label_to_index_bert_for_' + dataname + '_' + fs_method + '_' + num_of_features + '.txt', 'w', encoding='utf-8') as f:
    #     for i in label2ind.keys():
    #         f.writelines(str(i) + ':' + str(label2ind[i]) + '\n')
    #     f.close()
    labels = [label2ind[label] for label in labels]
    # print(samples[0])
    # print(labels[0])

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
    # sheet.write(0, 9, 'avg_of_recall')
    # sheet.write(0, 10, 'std_of_recall')
    # sheet.write(0, 11, 'avg_of_training_time')
    # sheet.write(0, 12, 'std_of_training_time')
    # sheet.write(0, 13, 'avg_of_test_time')
    # sheet.write(0, 14, 'std_of_test_time')
    sheet.write(0, 9, 'avg_of_training_time')
    sheet.write(0, 10, 'std_of_training_time')
    sheet.write(0, 11, 'avg_of_test_time')
    sheet.write(0, 12, 'std_of_test_time')
    samples_train_list, labels_train_list, samples_eval_list, labels_eval_list = k_fold_cross_validation(
        samples, labels, k)

    num_of_filters = 100  # 卷积核个数
    window_sizes = [2, 3, 4]  # 卷积核尺寸
    hidden_size_lstm = 100  # lstm的隐藏层尺寸
    classifier_list = ['basebert', 'bert+wa', 'bert+cnn', 'bert+lstm', 'bert+lstm+att', 'bert+lstm+cnn']
    # classifier_list = ['bert+lstm+att']
    # measures_total = np.zeros((k, len(classifier_list), 7))
    measures_total = np.zeros((k, len(classifier_list), 6))
    for i in range(k):
        # print('第%s折' % i)
        dataset_train = Data.DataLoader(dataset=MyDataset(samples_train_list[i], labels_train_list[i]), batch_size=batch_size_training, shuffle=True, num_workers=0)
        dataset_eval = Data.DataLoader(dataset=MyDataset(samples_eval_list[i], labels_eval_list[i]), batch_size=batch_size_test, shuffle=True, num_workers=0)
        for j in range(len(classifier_list)):
            # print(classifier_list[j])
            bc = 0
            if classifier_list[j] == 'basebert':
                bc = BaseBert().to(device)
                optimizer = optim.Adam(bc.parameters(), lr=1e-4, weight_decay=1e-2)
            elif classifier_list[j] == 'bert+wa':
                bc = BertWA().to(device)
                optimizer = optim.Adam(bc.parameters(), lr=1e-5, weight_decay=1e-2)
            elif classifier_list[j] == 'bert+cnn':
                bc = BertCNN().to(device)
                optimizer = optim.Adam(bc.parameters(), lr=1e-4, weight_decay=1e-2)
            elif classifier_list[j] == 'bert+lstm':
                bc = BertLSTM().to(device)
                optimizer = optim.Adam(bc.parameters(), lr=1e-4, weight_decay=1e-2)
                # optimizer = optim.SGD(bc.parameters(), lr=1e-7)
            elif classifier_list[j] == 'bert+lstm+att':
                bc = BertLSTMAttention().to(device)
                optimizer = optim.Adam(bc.parameters(), lr=1e-5, weight_decay=1e-2)
            elif classifier_list[j] == 'bert+lstm+cnn':
                bc = BertLSTMCNN().to(device)
                optimizer = optim.Adam(bc.parameters(), lr=1e-4, weight_decay=1e-2)

            # optimizer = optim.Adam(bc.parameters(), lr=1e-2)
            loss_fn = nn.CrossEntropyLoss().cuda()
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
                print(
                    '第{}折，共{}折，分类器为{}，第{}期，共{}期，损失为{}'.format(i + 1, k, classifier_list[j], epoch + 1, epoches, sumloss))
                # if epoch % 10 == 0:
                #     print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch + 1, epoches, i + 1, total_step, sumloss))
                # acc, auc, mavg, mae, recall = eval()
                acc, auc, mavg, mae = eval()
                if acc > best_val_acc:
                    torch.save(bc.state_dict(),
                               'model/model_params_best_{data}_{samplenum}_{fsmethod}_{fsnum}_{modelname}_{k_num}.pkl'.format(
                                   data=dataname, samplenum=num_of_sample, fsmethod=fs_method, fsnum=num_of_features,
                                   modelname=classifier_list[j], k_num=i))
                    # torch.save(bc.state_dict(),
                    #            'model/model_params_best_{data}_{fsmethod}_{fsnum}_{modelname}_{k_num}.pkl'.format(
                    #                data=dataname, fsmethod=fs_method, fsnum=num_of_features, modelname=classifier_list[j], k_num=i))
                    best_val_acc = acc
                    best_val_auc = auc
                    best_val_mavg = mavg
                    best_val_mae = mae
                    # best_val_recall = recall
                    bestepoch = epoch
                    trainingend = time.time()
            print('for k = %s' % i, 'bestepoch:', bestepoch, 'bestacc:', best_val_acc)
            teststart = time.time()
            bc.load_state_dict(torch.load(
                'model/model_params_best_{data}_{samplenum}_{fsmethod}_{fsnum}_{modelname}_{k_num}.pkl'.format(
                    data=dataname, samplenum=num_of_sample, fsmethod=fs_method, fsnum=num_of_features,
                    modelname=classifier_list[j], k_num=i)))
            # bc.load_state_dict(torch.load(
            #     'model/model_params_best_{data}_{fsmethod}_{fsnum}_{modelname}_{k_num}.pkl'.format(
            #     data=dataname, fsmethod=fs_method, fsnum=num_of_features, modelname=classifier_list[j], k_num=i)))
            print('for model = {modelname}, k = {k_num}'.format(modelname=classifier_list[j],k_num=k), 'test best model:')
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
    # book.save('e:/pythonwork/newclassification/results/' + dataname + '_' + num_of_sample + '_bert_criteria.xls')


    # book.save(
    #     'e:/pythonwork/newclassification/results/' + dataname + '_bert_criteria.xls')
    book.save(
        'e:/pythonwork/newclassification/results_with_feature_selection/' + dataname + '_' + num_of_sample
        + '_' + fs_method + '_' + num_of_features + '_bert_criteria2.xls')
    #
