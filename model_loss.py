import glob
import logging
import shutil
import subprocess
import time

import numpy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.stats as ss
import math
import pandas as pd



class Qnetwork(nn.Module):
    def __init__(self, h_size, action_num):
        super(Qnetwork, self).__init__()
        # The network recieves a state from the sumo, flattened into an array.
        # It then resizes it and processes it through three convolutional layers.
        self.conv1 = nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        
        # 计算卷积层输出大小
        # 输入: 60x60x2
        # conv1: (60-4)/2 + 1 = 29
        # conv2: (29-2)/1 + 1 = 28
        # conv3: (28-2)/1 + 1 = 27
        # 输出: 27x27x128
        conv_output_size = 27 * 27 * 128
        
        # It is split into Value and Advantage
        self.stream0 = nn.Linear(conv_output_size, 128)
        
        self.streamA0 = nn.Linear(128, h_size)
        self.streamV0 = nn.Linear(128, h_size)
        
        self.AW = nn.Linear(h_size, action_num)
        self.VW = nn.Linear(h_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        # Reshape input
        x = x.view(-1, 2, 60, 60)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Split into streams
        x = F.relu(self.stream0(x))
        
        streamA = F.relu(self.streamA0(x))
        streamV = F.relu(self.streamV0(x))
        
        # Calculate Advantage and Value
        Advantage = self.AW(streamA)
        Advantage = (Advantage - Advantage.mean(dim=1, keepdim=True)) / (Advantage.std(dim=1, keepdim=True) + 1e-5)
        Value = self.VW(streamV)
        
        # Combine to get Q-values
        Qout = Value + (Advantage - Advantage.mean(dim=1, keepdim=True))
        
        return Qout, Advantage
