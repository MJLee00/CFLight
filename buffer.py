import glob
import logging
import shutil
import subprocess
import time

import numpy
import numpy as np
import random

import os
import scipy.stats as ss
import math
import pandas as pd


class priorized_experience_buffer():
    def __init__(self, buffer_size=20000):
        self.buffer = []
        self.prob = []
        self.err = []
        self.buffer_size = buffer_size
        self.alpha = 0.2
        self.batch_size = 128
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.err[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
            self.prob[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
        self.err.extend([10000] * len(experience))
        self.prob.extend([1] * len(experience))

    def updateErr(self, indx, error):
        for i in range(0, len(indx)):
            # 使用绝对值并确保值为正数
            error_value = abs(error[i])
            self.err[indx[i]] = math.sqrt(error_value)
        r_err = ss.rankdata(self.err)  # rank of the error from smallest (1) to largest
        self.prob = [1 / (len(r_err) - i + 1) for i in r_err]

    def priorized_sample(self, size):
        prb = [i ** self.alpha for i in self.prob]
        t_s = [prb[0]]
        for i in range(1, len(self.prob)):
            t_s.append(prb[i] + t_s[i - 1])
        batch = []
        mx_p = t_s[-1]

        smp_set = set()

        while len(smp_set) < self.batch_size:
            tmp = np.random.uniform(0, mx_p)
            for j in range(0, len(t_s)):
                if t_s[j] > tmp:
                    smp_set.add(max(j - 1, 0))
                    break
                    
        # 创建一个结构化数组来存储不同形状的数据
        dtype = [('experience', object), ('index', int)]
        batch = np.zeros(len(smp_set), dtype=dtype)
        
        for i, idx in enumerate(smp_set):
            batch[i] = (self.buffer[idx], idx)
            
        return batch
