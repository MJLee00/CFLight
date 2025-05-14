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
from model_loss import Qnetwork
import datetime
from gan_cf_loss import CTRLG



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
path = "./dqn_cflight"  # The path to save our model to.
h_size = 64  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network
DEFAULT_PORT = 8813
safe_weight = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


ctrl_g = CTRLG(state_dim=3600, lr_g=1e-4, lr_d=1e-3, action_dim=10, noise_dim_s=4, noise_dim_r=4, hidden_dim=600, reward_dim=2, batch_size=batch_size)


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
cf_buffer = priorized_experience_buffer()
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

def get_safe_action(collision_ph, default_ph):
    if collision_ph == 0 or collision_ph == 1 or collision_ph == 2:
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif collision_ph == 5 or collision_ph == 6 or collision_ph == 7:
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        return default_ph

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
            with torch.no_grad():
                Q, adv = mainQN(s.to(device))
                adv = adv.detach().cpu().numpy()
        else:
            with torch.no_grad():
                Q, adv = mainQN(s.to(device))
                adv = adv.detach().cpu().numpy()
                a = Q.max(1)[1].item()

        s1, r, d, wait_time, collision_ph, collisions, r_eff, cf_result = take_action_safe_loss(i, j, None, None, collisions, tls, a, wait_time, wait_time_map, safe_weight, net_type)

        collision_count += collisions
        total_steps += 1
        # 确保所有输入都是numpy数组并具有正确的形状
        s_np = s.numpy() if torch.is_tensor(s) else s
        s1_np = s1.numpy() if torch.is_tensor(s1) else s1
        desired_action_distribution = get_safe_action(collision_ph, adv)
        myBuffer0.add(np.reshape(np.array([s_np, a, r, s1_np, d, collisions, r_eff, desired_action_distribution], dtype=object),
                                      [1, 8]))
        
        if total_steps > pre_train_steps:
            trainBatch = myBuffer0.priorized_sample(batch_size)
            if i > 50:
                cf_trainBatch = cf_buffer.priorized_sample(batch_size)
                batch_size_tmp = len(trainBatch) + len(cf_trainBatch)
                trainBatch = np.concatenate([trainBatch, cf_trainBatch], axis=0)
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
                desired_action_distribution = np.vstack(trainBatch[:, 7])
                # 假设rewards、end_multiplier是二维的object数组，先转为一维float数组
                rewards = np.array([float(r) for r in rewards.flatten()])
                actions = np.array([int(a) for a in actions.flatten()]) 
                end_multiplier = np.array([float(e) for e in dones.flatten()])
                
                # 确保next_states的形状正确
                if isinstance(next_states[0], np.ndarray):
                    next_states = np.stack(next_states)
                else:
                    next_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in next_states])
                Q, _ = mainQN(torch.FloatTensor(next_states).to(device))
                Q1 = Q.max(1)[1]
                # Get Q values from target network
                Q2, _ = targetQN(torch.FloatTensor(next_states).to(device))
          
                # Get target Q values
                if i > 50:
                    doubleQ = Q2[range(batch_size_tmp), Q1]
                else:
                    doubleQ = Q2[range(batch_size), Q1]
                #doubleQ = Q2[range(batch_size), Q1]
                targetQ = torch.FloatTensor(rewards).to(device) + (y * doubleQ * torch.FloatTensor(end_multiplier).to(device))

                # Update main network
                optimizer.zero_grad()
                current_states = trainBatch[:, 0]
                # 确保current_states的形状正确
                if isinstance(current_states[0], np.ndarray):
                    current_states = np.stack(current_states)
                else:
                    current_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in current_states])
                current_Q, Adv = mainQN(torch.FloatTensor(current_states).to(device))
                current_Q = current_Q.gather(1, torch.LongTensor(actions).unsqueeze(1).to(device))
                safe_loss = nn.CrossEntropyLoss()(Adv, torch.FloatTensor(desired_action_distribution).to(device))
                loss = nn.MSELoss()(current_Q, targetQ.unsqueeze(1).to(device)) 
                safe_loss = safe_loss * torch.FloatTensor(collisions).to(device).mean() * 10
                loss = loss + safe_loss 
                loss.backward()
                optimizer.step()
                td_error = (current_Q - targetQ.unsqueeze(1).to(device)).detach().cpu().numpy()
            

                if i > 50:
                    myBuffer0.updateErr(indices[:batch_size], td_error[:batch_size,:])
                    cf_buffer.updateErr(indices[batch_size:], td_error[batch_size:,:])
                else:
                    myBuffer0.updateErr(indices, td_error)

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
        df.to_csv('CFLight_loss_sync.csv')
    elif net_type == 0:
        df.to_csv('CFLight_loss_real.csv')
    
    end()
    
    if i % 50 == 0:
        cf_buffer.buffer.clear(), cf_buffer.err.clear(), cf_buffer.prob.clear()

        coll = np.concatenate([np.array(item[5]).reshape(1) for item in myBuffer0.buffer], axis=0)
        col_index = np.where(coll > 0)[0]
        state = np.concatenate([item[0] for item in myBuffer0.buffer], axis=0)[:,:,1]
        action = np.concatenate([np.array(item[1]).reshape(1) for item in myBuffer0.buffer], axis=0)
        reward = np.concatenate([np.array(item[2]).reshape(1) for item in myBuffer0.buffer], axis=0)
        next_state = np.concatenate([item[3] for item in myBuffer0.buffer], axis=0)[:,:,1]
        r_eff = np.concatenate([np.array(item[6]).reshape(1) for item in myBuffer0.buffer], axis=0)
        coll = coll[col_index]
        state = state[col_index]
        action = action[col_index]
        reward = reward[col_index]
        next_state = next_state[col_index]
        r_eff = r_eff[col_index]

        data_tmp = np.concatenate([state, action.reshape(-1,1), next_state, coll.reshape(-1,1), r_eff.reshape(-1,1)], axis=1)
        data_tmp = torch.tensor(data_tmp, dtype=torch.float32).to(device)
        ctrl_g.train_bicogan(data_tmp, epochs=500, batch_size=ctrl_g.batch_size)
        for j in range(data_tmp.shape[0]):
            augmented_data = ctrl_g.generate_counterfactuals(data_tmp[j].view(1, -1), coll[j], action_space=10)
            if action[j] == 0 or action[j] == 1 or action[j] == 2:
                desired_action_distribution = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif action[j] == 5 or action[j] == 6 or action[j] == 7:
                desired_action_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            else:
                desired_action_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                desired_action_distribution[action[j]] = 1
            for i in range(len(augmented_data)):
                state_aug = torch.zeros(1, ctrl_g.state_dim, 2)
                state_aug[:,:,0] = 1
                state_aug[:,:,1] = augmented_data[i][:ctrl_g.state_dim]
                action_aug = augmented_data[i][ctrl_g.state_dim:ctrl_g.state_dim+1]
                next_state_aug = torch.zeros(1, ctrl_g.state_dim, 2)
                next_state_aug[:,:,0] = 1
                next_state_aug[:,:,1] = augmented_data[i][ctrl_g.state_dim+1:ctrl_g.state_dim+1+ctrl_g.state_dim]
                coll_aug = augmented_data[i][ctrl_g.state_dim+1+ctrl_g.state_dim:ctrl_g.state_dim+1+ctrl_g.state_dim+1]
                r_eff_aug = augmented_data[i][ctrl_g.state_dim+1+ctrl_g.state_dim+1:]
                reward_aug = (r_eff_aug).detach().cpu().numpy().astype(float)
                state_aug = state_aug.numpy()
                action_aug = action_aug.detach().cpu().numpy().astype(int)
                next_state_aug = next_state_aug.numpy()
                coll_aug = coll_aug.detach().cpu().numpy().astype(float)
                r_eff_aug = r_eff_aug.detach().cpu().numpy().astype(float)
                
                #desired_action_distribution = get_safe_action(coll_aug, adv)
                cf_buffer.add(np.reshape(np.array([state_aug, action_aug, reward_aug, next_state_aug, False, coll_aug, r_eff_aug, desired_action_distribution], dtype=object),
                                            [1, 8]))


            
        
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
