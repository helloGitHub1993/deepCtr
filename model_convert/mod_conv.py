#!/usr/bin/env python
# -*- coding=utf-8 -*-

import sys

#tensor in model
tensor_name = {"embedding_ret_num","embedding_cmt_num","embedding_like_num","embedding_act_num","embedding_interact_num",\
		"embedding_expo_num","embedding_recent_ret_num","embedding_recent_cmt_num","embedding_recent_like_num",\
		"embedding_recent_act_num","embedding_recent_real_expo_num","embedding_effect_weight","embedding_author_followers_num",\
		"embedding_click_rate","embedding_interact_rate","embedding_click_rate_recent","embedding_interact_rate_recent",\
		"Variable","Variable_1","Variable_2","Variable_3","Variable_4","Variable_5",\
		"fullc_0/kernel","fullc_0/bias","fullc_1/kernel","fullc_1/bias","fullc_3/kernel","fullc_3/bias"}

tensor_name_list = ["embedding_ret_num","embedding_cmt_num","embedding_like_num","embedding_act_num","embedding_interact_num",\
			"embedding_expo_num","embedding_recent_ret_num","embedding_recent_cmt_num","embedding_recent_like_num",\
			"embedding_recent_act_num","embedding_recent_real_expo_num","embedding_effect_weight","embedding_author_followers_num",\
			"embedding_click_rate","embedding_interact_rate","embedding_click_rate_recent","embedding_interact_rate_recent",\
			"Variable","Variable_1","Variable_2","Variable_3","Variable_4","Variable_5",\
			"fullc_0/kernel","fullc_0/bias","fullc_1/kernel","fullc_1/bias","fullc_3/kernel","fullc_3/bias"]

para_name=""
tmp_dt=""
res = {}
for line in open('./mod'):
	if "tensor_name" in line:
		tmp = line.lstrip('\n').split(':')
		if para_name != "":
			res[para_name] = tmp_dt.strip('|')
			tmp_dt=""
			para_name=""
		if tmp[1].strip() in tensor_name:
			para_name = tmp[1].strip()
			
	elif para_name != "" and "[" in line:
		if "]" in line:
			dt = line.strip().strip('[').strip('\n').strip(']').strip()
			tmp_dt += dt + '|'
		else:	
			dt = line.strip().strip('[').strip('\n').strip()
			tmp_dt += dt
	elif para_name != "" and "]" in line:
		dt = line.strip('\n').strip(']').strip()
		tmp_dt += ' ' + dt + '|'
	elif para_name != "":
		dt = line.strip('\n').strip()
		tmp_dt += ' ' + dt

if para_name != "":
	res[para_name] = tmp_dt.strip('|')
i = 0
for key in tensor_name_list:
	if key in res.keys():
		val = res[key]
		val = val.strip('\n').split('|')
		#print(key)
		#print("start:" + str(i))
		for dt in val:
			dt = dt.strip('\n').split(' ')
			for final_val in dt:
				if final_val != "":
					print(str(i) + ":" + str('{:.10f}'.format(float(final_val))))
					i += 1
		 
		#print("end:" + str(i))
