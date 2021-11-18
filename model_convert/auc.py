#!/usr/bin/env python
#coding=utf-8

import sys
from sklearn import metrics

ckpt_pre = []
label = []
mod_pre = []
for line in open('../ckpt_pre'):
	line = line.strip('\n').split(',')
	for val in line:
		ckpt_pre.append(float(val))


for line in open('../data/5'):
	line = line.strip('\n').split('\t')
	label.append(int(line[1]))

for line in open('./mod_pre'):
	line = line.strip('\n')
	mod_pre.append(float(line[0]))

fpr,tpr,thresholds = metrics.roc_curve(label,ckpt_pre,pos_label=1)
print("ckpt auc:",metrics.auc(fpr,tpr))

fpr,tpr,thresholds = metrics.roc_curve(label,mod_pre,pos_label=1)
print("mod auc:",metrics.auc(fpr,tpr))



