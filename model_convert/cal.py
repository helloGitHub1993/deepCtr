#!/usr/bin/env python
#coding=utf-8

import sys
import numpy as np

model_size = 4181
model_weight = []
embedding_ret_num = []
embedding_cmt_num = []
embedding_like_num = []
embedding_act_num = []
embedding_interact_num = []
embedding_expo_num = []
embedding_recent_ret_num = []
embedding_recent_cmt_num = []
embedding_recent_like_num = []
embedding_recent_act_num = []
embedding_recent_real_expo_num = []
embedding_effect_weight = []
embedding_author_followers_num = []
embedding_click_rate = []
embedding_interact_rate = []
embedding_click_rate_recent = []
embedding_interact_rate_recent = []
cross_weight_1 = []
cross_bias_1 = []
cross_weight_2 = []
cross_bias_2 = []
dense_0_weight = []
dense_0_bias = []
dense_1_weight = []
dense_1_bias = []
dense_2_weight = []
dense_2_bias = []
libsvm_length = 8
embedding_size = 4
embed_layer_size = libsvm_length * embedding_size
feature_size = 68
fc_0_size = 32
fc_1_size = 32
dense_2_length = 1

def push_2d_data(embedding,size1,size2,start):
	if size1 >= 0 and size2 >= 0:
		for i in range(size1):
			tmp = []
			for j in range(size2):
				idx = i*size2 + j + start
				if idx < model_size:
					tmp.append(model_weight[idx])
			embedding.append(tmp)

def push_1d_data(embedding,start,end):
	if start < end:
		for i in range(start,end):
			if i < model_size:
				embedding.append(model_weight[i])



for line in open('./dcn_model_weight_final'):
	tmp = line.strip('\n').split(':')
	model_weight.append(float(tmp[1]))


push_2d_data(embedding_ret_num,libsvm_length,embedding_size,0)
push_2d_data(embedding_cmt_num,libsvm_length,embedding_size,embed_layer_size)
push_2d_data(embedding_like_num,libsvm_length,embedding_size,2*embed_layer_size)
push_2d_data(embedding_act_num,libsvm_length,embedding_size,3*embed_layer_size)
push_2d_data(embedding_interact_num,libsvm_length,embedding_size,4*embed_layer_size)
push_2d_data(embedding_expo_num,libsvm_length,embedding_size,5*embed_layer_size)
push_2d_data(embedding_recent_ret_num,libsvm_length,embedding_size,6*embed_layer_size)
push_2d_data(embedding_recent_cmt_num,libsvm_length,embedding_size,7*embed_layer_size)
push_2d_data(embedding_recent_like_num,libsvm_length,embedding_size,8*embed_layer_size)
push_2d_data(embedding_recent_act_num,libsvm_length,embedding_size,9*embed_layer_size)
push_2d_data(embedding_recent_real_expo_num,libsvm_length,embedding_size,10*embed_layer_size)
push_2d_data(embedding_effect_weight,libsvm_length,embedding_size,11*embed_layer_size)
push_2d_data(embedding_author_followers_num,libsvm_length,embedding_size,12*embed_layer_size)
push_2d_data(embedding_click_rate,libsvm_length,embedding_size,13*embed_layer_size)
push_2d_data(embedding_interact_rate,libsvm_length,embedding_size,14*embed_layer_size)
push_2d_data(embedding_click_rate_recent,libsvm_length,embedding_size,15*embed_layer_size)
push_2d_data(embedding_interact_rate_recent,libsvm_length,embedding_size,16*embed_layer_size)


start = 0 
end = 0
#cross layer
start = 17*embed_layer_size
end = start + feature_size
push_1d_data(cross_weight_1,start,end)

start = end
end += feature_size
push_1d_data(cross_bias_1,start,end)

start = end
end += feature_size
push_1d_data(cross_weight_2,start,end)

start = end
end += feature_size
push_1d_data(cross_bias_2,start,end)


#fc layer
start = end
end += feature_size*fc_0_size
push_2d_data(dense_0_weight,feature_size,fc_0_size,start)

start = end
end += fc_0_size
push_1d_data(dense_0_bias,start,end)

start = end
end += fc_0_size*fc_1_size
push_2d_data(dense_1_weight,fc_0_size,fc_1_size,start)

start = end
end += fc_1_size
push_1d_data(dense_1_bias,start,end)

start = end
end += fc_1_size + feature_size
push_1d_data(dense_2_weight,start,end) 

start = end
end += dense_2_length
push_1d_data(dense_2_bias,start,end)

#cross layer
cross_weight_1 = np.array(cross_weight_1)
cross_bias_1 = np.array(cross_bias_1)
cross_weight_2 = np.array(cross_weight_2)
cross_bias_2 = np.array(cross_bias_2)

dense_0_weight = np.array(dense_0_weight)
dense_0_bias = np.array(dense_0_bias)
dense_1_weight = np.array(dense_1_weight)
dense_1_bias = np.array(dense_1_bias)
dense_2_weight = np.array(dense_2_weight)
dense_2_bias = np.array(dense_2_bias)

inpt = []
for line in open('../data/5'):
	line = line.strip('\n').split('\t')
	tmp = line[0].strip().split(',')
	vec = []
	for i in range(len(tmp)):
		vec.append(int(tmp[i]))
	inpt.append(vec)

mk = 0
for feature in inpt:
	libsvm = []
	libsvm += embedding_ret_num[feature[0]]
	libsvm += embedding_cmt_num[feature[1]]
	libsvm += embedding_like_num[feature[2]]
	libsvm += embedding_act_num[feature[3]]
	libsvm += embedding_interact_num[feature[4]]
	libsvm += embedding_expo_num[feature[5]]
	libsvm += embedding_recent_ret_num[feature[6]]
	libsvm += embedding_recent_cmt_num[feature[7]]
	libsvm += embedding_recent_like_num[feature[8]]
	libsvm += embedding_recent_act_num[feature[9]]
	libsvm += embedding_recent_real_expo_num[feature[10]]
	libsvm += embedding_effect_weight[feature[11]]
	libsvm += embedding_author_followers_num[feature[12]]
	libsvm += embedding_click_rate[feature[13]]
	libsvm += embedding_interact_rate[feature[14]]
	libsvm += embedding_click_rate_recent[feature[15]]
	libsvm += embedding_interact_rate_recent[feature[16]]
	libsvm = np.array(libsvm)
	#cross layer
	c_1 = libsvm*(libsvm.dot(cross_weight_1.T))+cross_bias_1+libsvm
	c_2 = libsvm*(c_1.dot(cross_weight_2.T)) + cross_bias_2 + c_1

	#fc layer
	fc1 = 1/(1 + np.exp((libsvm.dot(dense_0_weight) + dense_0_bias)*(-1)))
	fc2 = 1/(1 + np.exp((fc1.dot(dense_1_weight) + dense_1_bias)*(-1)))
	print dense_0_weight[0]

	res =  np.concatenate((fc2,c_2),axis=0)
	
	print 1/(1 + np.exp((res.dot(dense_2_weight) + dense_2_bias)*(-1)))







