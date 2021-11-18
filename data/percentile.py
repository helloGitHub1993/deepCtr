#!/usr/bin/env python
# -*- coding=utf-8 -*-

import sys
import numpy as np


mk = 0
arr=[[] for i in range(10)]
net_type_set =set([])
born_set = set([])
gender_set = set([])
author_type_set = set([])
relation_set = set([])
recommend_source_set = set([])
for line in sys.stdin:
	line = line.strip("\n").split('\t')
	if len(line) != 17:
		continue
	arr[0].append(float(line[4]))
	arr[1].append(float(line[5]))
	if line[1] not in net_type_set:
		net_type_set.add(line[1])
	if line[2] not in born_set:
                born_set.add(line[2])
	if line[3] not in gender_set:
                gender_set.add(line[3])	
	if line[6] not in author_type_set:
                author_type_set.add(line[6])
	if line[15] not in relation_set:
                relation_set.add(line[15])
	if line[16] not in recommend_source_set:
                recommend_source_set.add(line[16])

	for i in range(7,15):
		arr[i-5].append(float(line[i]))
	mk += 1
	if mk >= 5000000:
		break

print(net_type_set)
print(born_set)
print(gender_set)
print(author_type_set)
print(relation_set)
print(recommend_source_set)

for j in range(len(arr)):
	res = np.array((arr[j]))
	print(np.percentile(res,[20,40,50,60,70,80,90]))

