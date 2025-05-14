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
from gan_cf import CTRLG
net_type = 1

# The parameters
batch_size = 128
update_freq = 4
y = .99
startE = 0.1
endE = 0.01
anneling_steps = 10000.
num_episodes = 800
pre_train_steps = 500
max_epLength = 500
load_model = False
action_num = 10
path = "./dqn_cflight_safe"
h_size = 64
tau = 0.001
DEFAULT_PORT = 8813
safe_weight = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Network configuration
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
    phases = ["GGGgrrrrGGGgrrrr", "yyygrrrryyygrrrr", "rrrGrrrrrrrGrrrr", "rrryrrrrrrryrrrr",
              "rrrrGGGgrrrrGGGg", "rrrryyygrrrryyyg", "rrrrrrrGrrrrrrrG", "rrrrrrryrrrrrrry"]
    left_turn_lanes = ['23429231#1_1', '28198821#3_1', '27115123#3_1', '-32038056#3_1']
    opposing_lanes = {
        '23429231#1_1': ['27115123#3_0', '27115123#3_1'],
        '28198821#3_1': ['-32038056#3_0', '-32038056#3_1'],
        '27115123#3_1': ['23429231#1_1', '23429231#1_0'],
        '-32038056#3_1': ['28198821#3_0', '28198821#3_1']
    }

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

# Initialize two Q-networks: one for effective reward, one for safety
mainQN_eff = Qnetwork(h_size, action_num).to(device)
targetQN_eff = Qnetwork(h_size, action_num).to(device)
targetQN_eff.load_state_dict(mainQN_eff.state_dict())

mainQN_safe = Qnetwork(h_size, action_num).to(device)
targetQN_safe = Qnetwork(h_size, action_num).to(device)
targetQN_safe.load_state_dict(mainQN_safe.state_dict())

# Initialize optimizers for both networks
optimizer_eff = optim.Adam(mainQN_eff.parameters(), lr=0.0001)
optimizer_safe = optim.Adam(mainQN_safe.parameters(), lr=0.0001)

# Initialize memory
myBuffer0 = priorized_experience_buffer()
cf_buffer = priorized_experience_buffer()
# Set the rate of random action decrease
e = startE
stepDrop = (startE - endE) / anneling_steps

# Lists to store episode data
jList = []
rList = []
wList = []
awList = []
tList = []
nList = []
data = []
total_steps = 0

# Make a path for model saving
if not os.path.exists(path):
    os.makedirs(path)

if load_model:
    print('Loading Model...')
    checkpoint = torch.load(os.path.join(path, 'model.pth'))
    mainQN_eff.load_state_dict(checkpoint['model_eff_state_dict'])
    targetQN_eff.load_state_dict(checkpoint['target_eff_state_dict'])
    mainQN_safe.load_state_dict(checkpoint['model_safe_state_dict'])
    targetQN_safe.load_state_dict(checkpoint['target_safe_state_dict'])
    optimizer_eff.load_state_dict(checkpoint['optimizer_eff_state_dict'])
    optimizer_safe.load_state_dict(checkpoint['optimizer_safe_state_dict'])

ctrl_g = CTRLG(state_dim=3600, lr_g=1e-4, lr_d=1e-3, action_dim=10, noise_dim_s=4, noise_dim_r=4, hidden_dim=600, reward_dim=2, batch_size=batch_size)

def updateTargetGraph(mainQN, targetQN, tau):
    for target_param, main_param in zip(targetQN.parameters(), mainQN.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

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
                Q_eff, adv_eff = mainQN_eff(s.to(device))
                Q_safe, adv_safe = mainQN_safe(s.to(device))
                adv_eff = adv_eff.detach().cpu().numpy()
        else:
            with torch.no_grad():
                Q_eff, _ = mainQN_eff(s.to(device))
                Q_safe, _ = mainQN_safe(s.to(device))
                # Combine Q-values with 0.5 weight for action selection
                combined_Q = 0.5 * Q_eff + 0.5 * Q_safe
                a = combined_Q.max(1)[1].item()

        s1, r, d, wait_time, collision_ph, collisions, r_eff, cf_result = take_action_q_loss(i, j, None, None, collisions, tls, a, wait_time, wait_time_map, safe_weight, net_type)
        collisions = -collisions/safe_weight
        collision_count += -collisions*safe_weight
        total_steps += 1
        s_np = s.numpy() if torch.is_tensor(s) else s
        s1_np = s1.numpy() if torch.is_tensor(s1) else s1
 
        myBuffer0.add(np.reshape(np.array([s_np, a, r, s1_np, d, collisions, r_eff], dtype=object), [1, 7]))

        if total_steps > pre_train_steps:
            trainBatch = myBuffer0.priorized_sample(batch_size)
            if i > 50:
                cf_trainBatch = cf_buffer.priorized_sample(batch_size)
                batch_size_tmp = len(trainBatch) + len(cf_trainBatch)
                trainBatch = np.concatenate([trainBatch, cf_trainBatch], axis=0)

            experiences = trainBatch['experience']
            indices = trainBatch['index']
            trainBatch = np.array([exp for exp in experiences])
            if e > endE:
                e -= stepDrop
            if total_steps % update_freq == 0:
                actions = np.vstack(trainBatch[:, 1])
                rewards = np.vstack(trainBatch[:, 2])
                next_states = np.vstack(trainBatch[:, 3])
                dones = np.vstack(trainBatch[:, 4])
                collisions = np.vstack(trainBatch[:, 5])
                r_eff = np.vstack(trainBatch[:, 6])
          
                rewards = np.array([float(r) for r in rewards.flatten()])
                r_eff = np.array([float(r) for r in r_eff.flatten()])
                actions = np.array([int(a) for a in actions.flatten()])
                end_multiplier = np.array([float(e) for e in dones.flatten()])
                collision_rewards = np.array([float(c) for c in collisions.flatten()])
   
                if isinstance(next_states[0], np.ndarray):
                    next_states = np.stack(next_states)
                else:
                    next_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in next_states])

                # Training for effective reward network
                Q_eff, _ = mainQN_eff(torch.FloatTensor(next_states).to(device))
                Q1_eff = Q_eff.max(1)[1]
                Q2_eff, _ = targetQN_eff(torch.FloatTensor(next_states).to(device))

                if i > 50:
                    doubleQ_eff = Q2_eff[range(batch_size_tmp), Q1_eff]
                else:
                    doubleQ_eff = Q2_eff[range(batch_size), Q1_eff]

        
                targetQ_eff = torch.FloatTensor(r_eff).to(device) + (y * doubleQ_eff * torch.FloatTensor(end_multiplier).to(device))

                optimizer_eff.zero_grad()
                current_states = trainBatch[:, 0]
                if isinstance(current_states[0], np.ndarray):
                    current_states = np.stack(current_states)
                else:
                    current_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in current_states])
                current_Q_eff, Adv_eff = mainQN_eff(torch.FloatTensor(current_states).to(device))
                current_Q_eff = current_Q_eff.gather(1, torch.LongTensor(actions).unsqueeze(1).to(device))
                loss_eff = nn.MSELoss()(current_Q_eff, targetQ_eff.unsqueeze(1).to(device))
                loss_eff.backward()
                optimizer_eff.step()
                td_error_eff = (current_Q_eff - targetQ_eff.unsqueeze(1).to(device)).detach().cpu().numpy()

                # Training for safety network
                Q_safe, _ = mainQN_safe(torch.FloatTensor(next_states).to(device))
                Q1_safe = Q_safe.max(1)[1]
                Q2_safe, _ = targetQN_safe(torch.FloatTensor(next_states).to(device))

                if i > 50:
                    doubleQ_safe = Q2_safe[range(batch_size_tmp), Q1_safe]
                else:
                    doubleQ_safe = Q2_safe[range(batch_size), Q1_safe]

             
                targetQ_safe = torch.FloatTensor(collision_rewards).to(device) 
                targetQ_safe += (y * doubleQ_safe * torch.FloatTensor(end_multiplier).to(device))

                optimizer_safe.zero_grad()
                current_Q_safe, Adv_safe = mainQN_safe(torch.FloatTensor(current_states).to(device))
                current_Q_safe = current_Q_safe.gather(1, torch.LongTensor(actions).unsqueeze(1).to(device))
                
                loss_safe = nn.MSELoss()(current_Q_safe, targetQ_safe.unsqueeze(1).to(device))
                loss_safe = loss_safe
                loss_safe.backward()
                optimizer_safe.step()
                td_error_safe = (current_Q_safe - targetQ_safe.unsqueeze(1).to(device)).detach().cpu().numpy()

                # Combine TD errors for buffer update (average for simplicity)
                td_error = 0.5 * td_error_eff + 0.5 * td_error_safe

                if i > 50:
                    myBuffer0.updateErr(indices[:batch_size], td_error[:batch_size,:])
                    cf_buffer.updateErr(indices[batch_size:], td_error[batch_size:,:])
                else:
                    myBuffer0.updateErr(indices, td_error)

                # Update target networks
                updateTargetGraph(mainQN_eff, targetQN_eff, tau)
                updateTargetGraph(mainQN_safe, targetQN_safe, tau)

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
        df.to_csv('CFLight_q_sync.csv')
    elif net_type == 0:
        df.to_csv('CFLight_q_real.csv')
    
    end()
     
     ##  CTC 算法
    if i % 50 == 0:
        cf_buffer.buffer.clear(), cf_buffer.err.clear(), cf_buffer.prob.clear()
        
        coll = np.concatenate([np.array(item[5]).reshape(1) for item in myBuffer0.buffer], axis=0)
        col_index = np.where(coll < 0)[0]
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

        data_tmp = np.concatenate([state, action.reshape(-1,1), next_state, -(coll.reshape(-1,1)/safe_weight), r_eff.reshape(-1,1)], axis=1)
        data_tmp = torch.tensor(data_tmp, dtype=torch.float32).to(device)
        ctrl_g.train_bicogan(data_tmp, epochs=500, batch_size=ctrl_g.batch_size)
        augmented_data = ctrl_g.generate_counterfactuals(data_tmp, -coll/safe_weight, action_space=10)
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
            cf_buffer.add(np.reshape(np.array([state_aug, action_aug, reward_aug, next_state_aug, False, coll_aug, r_eff_aug], dtype=object),
                                        [1, 7]))

        

    print("Total Reward---------------", rAll)
    if i % 100 == 0:
        torch.save({
            'model_eff_state_dict': mainQN_eff.state_dict(),
            'target_eff_state_dict': targetQN_eff.state_dict(),
            'model_safe_state_dict': mainQN_safe.state_dict(),
            'target_safe_state_dict': targetQN_safe.state_dict(),
            'optimizer_eff_state_dict': optimizer_eff.state_dict(),
            'optimizer_safe_state_dict': optimizer_safe.state_dict(),
        }, os.path.join(path, f'model-{i}.pth'))
        print("Saved Model")

torch.save({
    'model_eff_state_dict': mainQN_eff.state_dict(),
    'target_eff_state_dict': targetQN_eff.state_dict(),
    'model_safe_state_dict': mainQN_safe.state_dict(),
    'target_safe_state_dict': targetQN_safe.state_dict(),
    'optimizer_eff_state_dict': optimizer_eff.state_dict(),
    'optimizer_safe_state_dict': optimizer_safe.state_dict(),
}, os.path.join(path, f'model-{i}.pth'))
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")