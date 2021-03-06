#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

def dnn(input,dnn_hidden_units,dnn_activation=tf.nn.sigmoid,regularizer_rate=0.003,use_dropout=0,use_bn=False,seed=1024):
    """
    :param input: input, 输入，格式 bs * input_size
    :param dnn_hidden_units: dnn_hidden_units, list, 代表每层的hidden unit number
    :param dnn_activation:  dnn_activation
    :param regularizer_rate:
    :param use_dropout:
    :param use_bn:
    :param seed:
    :return:
    """
    for i in range(len(dnn_hidden_units)):
        input = tf.layers.dense(inputs=input,units=dnn_hidden_units[i],activation=dnn_activation,kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer_rate),name="fullc_{}".format(i))
        if use_bn:
            input = tf.layers.batch_normalization(inputs=input)
        if use_dropout != 0:
            input = tf.nn.dropout(input,use_dropout)

    return input


def cross(embeddings,layer_num=2):
    x_0 = tf.reshape(embeddings,[-1,embeddings.get_shape().as_list()[1],1])
    kernals,bias = {},{}
    glorot = np.sqrt(1.0 / (x_0.get_shape().as_list()[1]))
    for i in range(layer_num):
        kernals[i] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(x_0.get_shape().as_list()[1],1)),dtype=tf.float32)
        bias[i] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(x_0.get_shape().as_list()[1],1)),dtype=tf.float32)

    x_l = x_0
    for i in range(layer_num):
        xl_w = tf.tensordot(x_l,kernals[i],axes=(1,0))
        doc_ = tf.matmul(x_0,xl_w) 
        x_l = doc_ + bias[i] + x_l
    x_l = tf.squeeze(x_l,axis=2)

    return x_l


def get_embedding(feature_list,feature_dict,sparse_data,embedding_size,num_of_feature):
    embedding_dict = {}
    for val in feature_list:
        embedding_dict[val] = tf.get_variable("embedding_{}".format(val),[feature_dict[val],embedding_size])
    sparse_data_list = tf.split(sparse_data,num_of_feature,axis=1)
    sparse_data_embedding_list = []
    tmp = 0
    for val in feature_list:
        sparse_data_embedding_list.append(tf.nn.embedding_lookup(embedding_dict[val],sparse_data_list[tmp]))
        tmp += 1
    data_embedding = tf.concat(sparse_data_embedding_list,axis=1)
    return data_embedding

