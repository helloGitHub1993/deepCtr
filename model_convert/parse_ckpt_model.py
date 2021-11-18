#!/usr/bin/env python
# -*- coding=utf-8 -*-


import sys
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np

np.set_printoptions(threshold=np.inf)
chkp.print_tensors_in_checkpoint_file(file_name='../checkpoint/ckpt',tensor_name=None,all_tensors=True)

