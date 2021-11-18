#!/usr/bin/env python
# -*- coding=utf-8 -*-

import sys
import random
import time

#mk=0
for line in sys.stdin:
	tmp=line.strip('\n').split('\t')
	if len(tmp) != 3:
		continue
	if tmp[2] == "1":
		print(tmp[1] + '\t' + "1")
	else:
		random.seed(time.time())
		if random.random()>0.9:
			continue
		else:
			print(tmp[1] + '\t' + "0")
	#mk += 1
	#if mk >= 10:
	#	break
