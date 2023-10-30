from sklearn import metrics

y_matrix = [[0, 1], [1, 0]]
preds = [[0.3, 0.7], [0.8, 0.2]]
auc = metrics.roc_auc_score(y_matrix, preds, average='weighted')
print(auc)