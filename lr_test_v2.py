# coding=utf-8

from sklearn.linear_model import LinearRegression
from sklearn import metrics

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
            if len(libsvm) != 17:
                continue
            for i in range(len(libsvm)):
                data.append(int(libsvm[i]))
            final_data.append(data)
            final_label.append(label)
        return final_data,final_label

# 读取数据
X_train,y_train = _get_data('./data/train_dt_final_v2')
X_test,y_test = _get_data('./data/test_dt_final_v2')

clf = LinearRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
fpr,tpr,thresholds = metrics.roc_curve(y_test,pred,pos_label=1)
print(metrics.auc(fpr,tpr))
