#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import time
import datetime
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from model import Dcn
 
def _eval(sess,model):
    y_scores = []
    y_true = []
    epoch_num = 0
    epoch_size = round(len(test_data)/test_batch_size)
    while epoch_num < epoch_size:
        test_epoch_data = test_data[epoch_num*test_batch_size:min((epoch_num+1)*test_batch_size,len(test_data))]
        test_epoch_label = test_label[epoch_num*test_batch_size:min((epoch_num+1)*test_batch_size,len(test_label))]
        y_scores += list(model._eval(sess,test_epoch_data))
        y_true += list(test_epoch_label)
        epoch_num += 1
    y_scores = np.array(y_scores).flatten()
    y_true = np.array(y_true).flatten()
    test_gauc = roc_auc_score(y_true,y_scores)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        tmp = './checkpoint/ckpt'
        model.save(sess,tmp)
    print("Now auc is",test_gauc,"best auc is",best_auc)
    sys.stdout.flush()
    return None

def _eval_train(sess,model,epoch_num):
    y_scores = []
    y_true = []
    test_epoch_data = train_data_final[epoch_num*train_batch_size:min((epoch_num+1)*train_batch_size,len(train_data))]
    test_epoch_label = train_label_final[epoch_num*train_batch_size:min((epoch_num+1)*train_batch_size,len(train_data))]
    y_scores += list(model._eval(sess,test_epoch_data))
    y_true += list(test_epoch_label)
    y_scores = np.array(y_scores).flatten()
    y_true = np.array(y_true).flatten()
    test_gauc = roc_auc_score(y_true,y_scores)
    print("Now train auc is",test_gauc)
    sys.stdout.flush()
    return None

def _get_data(file):
    data = []
    with open(file,'r') as f:
        for line in f.readlines():
            res = line.strip('\n')
            data.append(res)
        return data

def _load_and_parse_data(file):
    final_data = []
    final_label = []
    with open(file,'r') as f:
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

def _parse_data(input):
    final_data = []
    final_label = []
    for i in range(len(input)):
        data = []
        tmp = input[i].strip('\n').split('\t')
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

random.seed(102)
np.random.seed(102)
best_auc = 0
train_batch_size = 1024
test_batch_size = 512

train_data = _get_data('./data/train_data')
test_data,test_label = _load_and_parse_data('./data/test_data')
new_test_data,_ = _load_and_parse_data('./data/5')

with tf.Session() as sess:
    model = Dcn()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    file_writer = tf.summary.FileWriter('./checkpoint', sess.graph)
    lr = 0.01
    start_time = time.time()
    if False:
        #reload model and predict
        tmp = './checkpoint/ckpt'
        model.restore(sess,tmp)
        res = list(model._eval(sess,new_test_data))
        print("pred:",res)
    else:
        #train model
        for _ in range(20):
            epoch_num = 0
            random.shuffle(train_data)
            train_data_final,train_label_final = _parse_data(train_data)
            epoch_size = round(len(train_data_final)/train_batch_size)
            while epoch_num < epoch_size:
                train_epoch_data = train_data_final[epoch_num*train_batch_size:min((epoch_num+1)*train_batch_size,len(train_data))]
                train_epoch_label = train_label_final[epoch_num*train_batch_size:min((epoch_num+1)*train_batch_size,len(train_data))]
                loss = model.train(sess,train_epoch_data,train_epoch_label,lr)
            
                if model.global_step.eval() % 200 == 0:
                    print("Now loss is:",loss)
                    sys.stdout.flush()
                    _eval(sess, model)
                    _eval_train(sess, model,epoch_num)

                if model.global_step.eval() % 10000 == 0:
                    lr *= 0.9

                epoch_num += 1

        print('Epoch %d DONE\tcost time:%.2f' %(model.global_epoch_step.eval(),time.time() - start_time))
        print("best auc is:",best_auc)

        sys.stdout.flush()
        model.global_epoch_step_op.eval()



