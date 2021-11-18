#!/usr/bin/env python
# -*- coding=utf-8 -*-
import sys
import json
import pickle
import time
import random

def bucket_num(num,bucket_list):
	low = 0
	high = len(bucket_list) - 1 
	while low <= high:
		middle = int((low + high)/2)
		if bucket_list[middle]>num:
			high = middle - 1 
		elif bucket_list[middle] < num:
			low = middle + 1
		else:
			return middle
	return low	


rule_dict = {}
#mk = 0
#load data conf file
for line in open('./data_conf','rb'):
	if "#" in line:
		continue
	if "bucketized_column" in line:
		rule = json.loads(line)
		idx =  rule["index"]
		if idx in rule_dict:
			continue
		rule_dict[idx] = rule["args"]

uid_mid_set = set([])
#for line in open('./train_data_original_v2','rb'):
#	tmp = line.strip('\n').split('\t')
#	if len(tmp) != 20:
#		continue
#	if tmp[15] == "1" or tmp[16] == "1" or tmp[17] == "1" or tmp[18] == "1" or tmp[19] == "1":
#		key = tmp[0] + ' ' + tmp[1]
#		uid_mid_set.add(key)

for line in sys.stdin:
	tmp = line.strip('\n').split('\t')
	res = ""
	if len(tmp) != 20:
		continue
	label = "0"
	if tmp[15] == "1" or tmp[16] == "1" or tmp[17] == "1" or tmp[18] == "1" or tmp[19] == "1":
		label = "1"
	key = tmp[0] + ' ' + tmp[1]
	if label == "0" and key in uid_mid_set:
		continue
	uid_mid_set.add(key)

	if label == "0":
		random.seed(time.time())
		if random.random() > 0.3:
			continue
	for i in range(2,15):
		idx_num = bucket_num(int(tmp[i]),rule_dict[str(i)])
		res += str(idx_num) + ","
	click_rate_idx=0
	interact_rate_idx=0
	click_rate_recent_idx=0
	interact_rate_recent_idx=0
	if tmp[7] != "0":
		click_rate=float(tmp[5])/float(tmp[7])
		interact_rate=float(tmp[6])/float(tmp[7])
		click_rate_idx = bucket_num(click_rate,rule_dict["15"])
		interact_rate_idx = bucket_num(interact_rate,rule_dict["16"])
	if tmp[12] != "0":
		click_rate_recent=float(tmp[11])/float(tmp[12])
		interact_rate_recent=(float(tmp[8])+float(tmp[9])+float(tmp[10]))/float(tmp[12])
		click_rate_recent_idx=bucket_num(click_rate_recent,rule_dict["17"])
		interact_rate_recent_idx=bucket_num(interact_rate_recent,rule_dict["18"])
	res += str(click_rate_idx) + ',' + str(interact_rate_idx) + ',' + str(click_rate_recent_idx) + ',' + str(interact_rate_recent_idx)
	#reweight click example
	if tmp[15] == "1":
		for i in range(2):
			print(key + '\t' + res.strip(',') + '\t' + label)
	#reweight follow  example
	elif tmp[19] == "1":
		for i in range(5):
			print(key + '\t' + res.strip(',') + '\t' + label)
	else:
		print(key + '\t' + res.strip(',') + '\t' + label)
	#mk += 1
	#if mk >= 10:
	#	break



