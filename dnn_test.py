# coding=utf-8

import numpy as np
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score as cv
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split as TTS
from sklearn import metrics
from sklearn.externals import joblib
from time import time
import datetime

def _get_data(file):
    final_data = []
    final_label = []
    with open(file,'rb') as f:
        for line in f.readlines():
            data = []
            tmp = line.strip('\n').split('\t')
            if len(tmp) != 2:
                continue
            label = int(tmp[1])
            libsvm = tmp[0].strip().split(',')
            if len(libsvm) != 16:
                continue
            for i in range(len(libsvm)):
                data.append(int(libsvm[i]))
            final_data.append(data)
            final_label.append(label)
        return final_data,final_label

# 读取数据
X_train,y_train = _get_data('./data/3')
X_test,y_test = _get_data('./data/4')


dnn = DNN(hidden_layer_sizes=(32,16),max_iter=100,random_state=2,activation='logistic')
dnn.fit(X_train,y_train)
#print(dnn.score(X_test,y_test))
#joblib.dump(dnn,'./dnn_model')
pred = dnn.predict(X_test)

fpr,tpr,thresholds = metrics.roc_curve(y_test,pred,pos_label=1)
print(metrics.roc_auc_score(y_test,pred))
