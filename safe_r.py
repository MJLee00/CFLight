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
import torch.optim as optim
import os
import scipy.stats as ss
import math
import pandas as pd
import math
import os, sys
from utils import *
from sumolib import checkBinary
import traci
import traci.constants as tc
import xml.etree.cElementTree as ET
import numpy as np
from buffer import priorized_experience_buffer
from model import Qnetwork
import datetime
from gan_cf import CTRLG



net_type = 1

# The parameters
batch_size = 128  # How many experiences to use for each training step.
update_freq = 4  # How often to perform a training step.
y = .99  # Discount factor on the target Q-values
startE = 0.1  # Starting chance of random action
endE = 0.01  # Final chance of random action
anneling_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 800  # 000 #How many episodes of game environment to train network with.
pre_train_steps = 500  # 0000 #How many steps of random actions before training begins.
max_epLength = 500  # The max allowed length of our episode.
load_model = False  # Whether to load a saved model.
action_num = 10  # total number of actions
path = "./dqn_r"  # The path to save our model to.
h_size = 64  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network
DEFAULT_PORT = 8813
safe_weight = 5
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


# TODO: 
# 路网1是sync，0是cologne
if net_type == 1:
    sumocfg_file = "cross/cross1.sumocfg"
    left_turn_lanes = ['4i_2', '2i_2', '3i_2', '1i_2']
    opposing_lanes = {
        '1i_2': ['2i_0', '2i_1'],
        '2i_2': ['1i_0', '1i_1'],
        '3i_2': ['4i_0', '4i_1'],
        '4i_2': ['3i_0', '3i_1']
    }
elif net_type == 0:
    sumocfg_file = "cologne1/cologne1.sumocfg"
    phases = ["GGGgrrrrGGGgrrrr", "yyygrrrryyygrrrr" "rrrGrrrrrrrGrrrr", "rrryrrrrrrryrrrr",
            "rrrrGGGgrrrrGGGg", "rrrryyygrrrryyyg", "rrrrrrrGrrrrrrrG", "rrrrrrryrrrrrrry"]
    left_turn_lanes = ['23429231#1_1', '28198821#3_1', '27115123#3_1', '-32038056#3_1']
    opposing_lanes = {
        '23429231#1_1': ['27115123#3_0', '27115123#3_1'],
        '28198821#3_1': ['-32038056#3_0', '-32038056#3_1'],
        '27115123#3_1': ['23429231#1_1', '23429231#1_0'],
        '-32038056#3_1': ['28198821#3_0', '28198821#3_1']
    }


ctrl_g = CTRLG(state_dim=3600, action_dim=1, noise_dim_s=4, noise_dim_r=4, hidden_dim=600, reward_dim=2, batch_size=batch_size)


def reset(is_analysis=False, sumocfg_file=sumocfg_file):
    if is_analysis:
        sumocfg_file = "cross/cross_wooutput.sumocfg"
    command = [checkBinary('sumo'), '-c', sumocfg_file, '--no-warnings', 'True']
    traci.start(command)

    tls = traci.trafficlight.getIDList()
    traci.trafficlight.setProgram(tls[0], '0')
    return tls

def end():
    traci.close()




# Initialize networks
mainQN = Qnetwork(h_size, action_num).to(device)
targetQN = Qnetwork(h_size, action_num).to(device)
targetQN.load_state_dict(mainQN.state_dict())

# Initialize optimizer
optimizer = optim.Adam(mainQN.parameters(), lr=0.0001)

# Initialize memory
myBuffer0 = priorized_experience_buffer()
# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / anneling_steps

# create lists to contain total rewards and steps per episode
jList = []  # number of steps in one episode
rList = []  # reward in one episode
wList = []  # the total waiting time in one episode
awList = []  # the average waiting time in one episode
tList = []  # thoughput in one episode (number of generated vehicles)
nList = []  # stops' percentage (number of stopped vehicles divided by the total generated vehicles)
data = []  # 用于存储训练日志的列表
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

if load_model:
    print('Loading Model...')
    checkpoint = torch.load(os.path.join(path, 'model.pth'))
    mainQN.load_state_dict(checkpoint['model_state_dict'])
    targetQN.load_state_dict(checkpoint['target_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for i in range(1, num_episodes):
    tls = reset()
    s = get_state()
    wait_time_map = {}
    wait_time = 0 
    collisions = 0
    d = False
    rAll = 0
    j = 0
    collision_count = 0
    while j < max_epLength:
        j += 1
        if (np.random.rand(1) < e or total_steps < pre_train_steps) and not load_model:
            a = np.random.randint(0, action_num)
        else:
            with torch.no_grad():
                a = mainQN(s.to(device)).max(1)[1].item()

        s1, r, d, wait_time, collision_ph, collisions, r_eff, cf_result = take_action(i, j, None, None, collisions, tls, a, wait_time, wait_time_map, safe_weight, net_type)

        collision_count += collisions
        total_steps += 1
        # 确保所有输入都是numpy数组并具有正确的形状
        s_np = s.numpy() if torch.is_tensor(s) else s
        s1_np = s1.numpy() if torch.is_tensor(s1) else s1
        myBuffer0.add(np.reshape(np.array([s_np, a, r, s1_np, d, collisions, r_eff], dtype=object),
                                      [1, 7]))

        if total_steps > pre_train_steps:
            trainBatch = myBuffer0.priorized_sample(batch_size)
    
            
            # 从结构化数组中提取经验数据
            experiences = trainBatch['experience']
            indices = trainBatch['index']
            
            # 将经验数据转换为numpy数组
            trainBatch = np.array([exp for exp in experiences])
            
            if e > endE:
                e -= stepDrop
            if total_steps % (update_freq) == 0:
 
            
                actions = np.vstack(trainBatch[:, 1])
                rewards = np.vstack(trainBatch[:, 2])
                next_states = np.vstack(trainBatch[:, 3])
                dones = np.vstack(trainBatch[:, 4])
                collisions = np.vstack(trainBatch[:, 5])
                
                # 假设rewards、end_multiplier是二维的object数组，先转为一维float数组
                rewards = np.array([float(r) for r in rewards.flatten()])
                actions = np.array([int(a) for a in actions.flatten()]) 
                end_multiplier = np.array([float(e) for e in dones.flatten()])
                
                # 确保next_states的形状正确
                if isinstance(next_states[0], np.ndarray):
                    next_states = np.stack(next_states)
                else:
                    next_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in next_states])
                Q1 = mainQN(torch.FloatTensor(next_states).to(device)).max(1)[1]
                
                # Get Q values from target network
                Q2 = targetQN(torch.FloatTensor(next_states).to(device))
                
                # Get target Q values
                doubleQ = Q2[range(batch_size), Q1]
                targetQ = torch.FloatTensor(rewards).to(device) + (y * doubleQ * torch.FloatTensor(end_multiplier).to(device))

                # Update main network
                optimizer.zero_grad()
                current_states = trainBatch[:, 0]
                # 确保current_states的形状正确
                if isinstance(current_states[0], np.ndarray):
                    current_states = np.stack(current_states)
                else:
                    current_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in current_states])
                current_Q = mainQN(torch.FloatTensor(current_states).to(device))
                current_Q = current_Q.gather(1, torch.LongTensor(actions).unsqueeze(1).to(device))
                loss = nn.MSELoss()(current_Q, targetQ.unsqueeze(1).to(device))
                loss.backward()
                optimizer.step()
                td_error = current_Q - targetQ.unsqueeze(1).to(device)
                myBuffer0.updateErr(indices, td_error.detach().cpu().numpy())
                # Update target network
                updateTargetGraph(mainQN, targetQN, tau)

        rAll += r
        s = s1

        if d == True:
            break

    jList.append(j)
    rList.append(rAll)
    wt = sum(wait_time_map[x] for x in wait_time_map)
    wtAve = wt / len(wait_time_map)
    wList.append(wtAve)
    awList.append(wt)
    tList.append(len(wait_time_map))
    tmp = [x for x in wait_time_map if wait_time_map[x] > 1]
    nList.append(len(tmp) / len(wait_time_map))

    log = {'collisions': (collision_count)/2,
           'step': total_steps,
           'episode_steps': j,
           'reward': rAll,
           'wait_average': wtAve,
           'accumulated_wait': wt,
           'throughput': len(wait_time_map),
           'stops': len(tmp) / len(wait_time_map)}
    data.append(log)
    logging.info('''Training: episode %d, total_steps: %d, sum_reward: %.2f, collisions: %d''' %
                 (i, total_steps, rAll, collision_count/2))
    df = pd.DataFrame(data)

    if net_type == 1:
        df.to_csv('safe_r_sync.csv')
    elif net_type == 0:
        df.to_csv('safe_r_real.csv')
    
    end()
    
        
    print("Total Reward---------------", rAll)
    # Periodically save the model
    if i % 100 == 0:
        torch.save({
            'model_state_dict': mainQN.state_dict(),
            'target_state_dict': targetQN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(path, f'model-{i}.pth'))
        print("Saved Model")

torch.save({
    'model_state_dict': mainQN.state_dict(),
    'target_state_dict': targetQN.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, os.path.join(path, f'model-{i}.pth'))
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
