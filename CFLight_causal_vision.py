import glob
import logging
import shutil
import subprocess
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import scipy.stats as ss
import math
import pandas as pd
from utils import *
from sumolib import checkBinary
import traci
import traci.constants as tc
import xml.etree.cElementTree as ET
from buffer import priorized_experience_buffer
from model import Qnetwork
import datetime
from gan_cf import CTRLG
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

net_type = 0

# The parameters
batch_size = 128
update_freq = 4
y = .99
startE = 0.1
endE = 0.01
anneling_steps = 10000.
num_episodes = 2
pre_train_steps = 500
max_epLength = 500
load_model = False
action_num = 10 
path = "./dqn_cflight_abalation"
h_size = 64
tau = 0.001
DEFAULT_PORT = 8813
safe_weight = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vis_path = os.path.join(path, "visualizations")
if not os.path.exists(vis_path):
    os.makedirs(vis_path)

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

mainQN = Qnetwork(h_size, action_num).to(device)
targetQN = Qnetwork(h_size, action_num).to(device)
targetQN.load_state_dict(mainQN.state_dict())

optimizer = optim.Adam(mainQN.parameters(), lr=0.0001)

myBuffer0 = priorized_experience_buffer()
cf_buffer = priorized_experience_buffer()

e = startE
stepDrop = (startE - endE) / anneling_steps

jList = []
rList = []
wList = []
awList = []
tList = []
nList = []
data = []
total_steps = 0

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
        s_np = s.numpy() if torch.is_tensor(s) else s
        s1_np = s1.numpy() if torch.is_tensor(s1) else s1
        myBuffer0.add(np.reshape(np.array([s_np, a, r, s1_np, d, collisions, r_eff], dtype=object), [1, 7]))

        if total_steps > pre_train_steps:
            trainBatch = myBuffer0.priorized_sample(batch_size)
            if i > 1:
                trainBatch = cf_buffer.priorized_sample(batch_size)
                batch_size_tmp = len(trainBatch)
            else:
                batch_size_tmp = batch_size

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

                rewards = np.array([float(r) for r in rewards.flatten()])
                actions = np.array([int(a) for a in actions.flatten()])
                end_multiplier = np.array([float(e) for e in dones.flatten()])

                if isinstance(next_states[0], np.ndarray):
                    next_states = np.stack(next_states)
                else:
                    next_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in next_states])
                Q1 = mainQN(torch.FloatTensor(next_states).to(device)).max(1)[1]
                Q2 = targetQN(torch.FloatTensor(next_states).to(device))
                doubleQ = Q2[range(batch_size_tmp), Q1]
                targetQ = torch.FloatTensor(rewards).to(device) + (y * doubleQ * torch.FloatTensor(end_multiplier).to(device))

                optimizer.zero_grad()
                current_states = trainBatch[:, 0]
                if isinstance(current_states[0], np.ndarray):
                    current_states = np.stack(current_states)
                else:
                    current_states = np.array([state.numpy() if torch.is_tensor(state) else state for state in current_states])
                current_Q = mainQN(torch.FloatTensor(current_states).to(device))
                current_Q = current_Q.gather(1, torch.LongTensor(actions).unsqueeze(1).to(device))
                loss = nn.MSELoss()(current_Q, targetQ.unsqueeze(1).to(device))
                loss.backward()
                optimizer.step()
                td_error = (current_Q - targetQ.unsqueeze(1).to(device)).detach().cpu().numpy()

                if i > 1:
                    cf_buffer.updateErr(indices, td_error)
                else:
                    myBuffer0.updateErr(indices, td_error)

                updateTargetGraph(mainQN, targetQN, tau)

        rAll += r
        s = s1

        if d == True:
            break

    jList.append(j)
    rList.append(rAll)
    wt = sum(wait_time_map[x] for x in wait_time_map)
    wtAve = wt / len(wait_time_map) if len(wait_time_map) > 0 else 0
    wList.append(wtAve)
    awList.append(wt)
    tList.append(len(wait_time_map))
    tmp = [x for x in wait_time_map if wait_time_map[x] > 1]
    nList.append(len(tmp) / len(wait_time_map) if len(wait_time_map) > 0 else 0)

    log = {
        'collisions': collision_count / 2,
        'step': total_steps,
        'episode_steps': j,
        'reward': rAll,
        'wait_average': wtAve,
        'accumulated_wait': wt,
        'throughput': len(wait_time_map),
        'stops': len(tmp) / len(wait_time_map) if len(wait_time_map) > 0 else 0
    }
    data.append(log)
    logging.info('''Training: episode %d, total_steps: %d, sum_reward: %.2f, collisions: %d''' %
                 (i, total_steps, rAll, collision_count / 2))
    df = pd.DataFrame(data)

    if net_type == 1:
        df.to_csv('CFLight_cfr_cf3dqn_sync.csv')
    elif net_type == 0:
        df.to_csv('CFLight_cfr_cf3dqn_real.csv')

    end()

    if i > 0:
        cf_buffer.buffer.clear(), cf_buffer.err.clear(), cf_buffer.prob.clear()

        coll = np.concatenate([np.array(item[5]).reshape(1) for item in myBuffer0.buffer], axis=0)
        col_index = np.where(coll > 0)[0]
        state = np.concatenate([item[0] for item in myBuffer0.buffer], axis=0)[:, :, 1]
        action = np.concatenate([np.array(item[1]).reshape(1) for item in myBuffer0.buffer], axis=0)
        reward = np.concatenate([np.array(item[2]).reshape(1) for item in myBuffer0.buffer], axis=0)
        next_state = np.concatenate([item[3] for item in myBuffer0.buffer], axis=0)[:, :, 1]
        r_eff = np.concatenate([np.array(item[6]).reshape(1) for item in myBuffer0.buffer], axis=0)
        
        data_tmp = np.concatenate([state, action.reshape(-1, 1), next_state, -(coll.reshape(-1, 1) / safe_weight), r_eff.reshape(-1, 1)], axis=1)
        data_tmp = torch.tensor(data_tmp, dtype=torch.float32).to(device)
        ctrl_g.train_bicogan(data_tmp, epochs=100, batch_size=ctrl_g.batch_size)

        coll = coll[col_index]
        state = state[col_index]
        action = action[col_index]
        reward = reward[col_index]
        next_state = next_state[col_index]
        r_eff = r_eff[col_index]

        for idx in range(min(len(state), 1)):
            state_vis = state[idx]
            state_vis = state_vis.reshape(60, 60)
            next_state_vis = next_state[idx]
            next_state_vis = next_state_vis.reshape(60, 60)
            a = action[idx]
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(state_vis, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Norm Speed')
            plt.title(f'Collision State')
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.subplot(1, 3, 2)
            plt.imshow(next_state_vis, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Norm Speed')
            plt.title(f'Collision Next State')
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.subplot(1, 3, 3)
            reward_map = np.zeros((60, 60))
            reward_map[:30, :] = -coll[idx]/safe_weight
            reward_map[30:, :] = r_eff[idx]
            plt.imshow(reward_map, cmap='coolwarm', interpolation='nearest', vmin=-2, vmax=2)
            plt.colorbar(label='Norm Reward')
            plt.title(f'Safe and Effeciency')
       
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path, f'action_{a}_sample_{idx}.pdf'))
            plt.close()
        data_tmp = data_tmp[col_index]
        augmented_data = ctrl_g.generate_counterfactuals(data_tmp[:1,:], -coll[:1] / safe_weight, action_space=10)
        for k in range(len(augmented_data)):    
            state_aug = torch.zeros(1, ctrl_g.state_dim, 2)
            state_aug[:, :, 0] = 1
            state_aug[:, :, 1] = augmented_data[k][:ctrl_g.state_dim]
            action_aug = augmented_data[k][ctrl_g.state_dim:ctrl_g.state_dim + 1]
            next_state_aug = torch.zeros(1, ctrl_g.state_dim, 2)
            next_state_aug[:, :, 0] = 1
            next_state_aug[:, :, 1] = augmented_data[k][ctrl_g.state_dim + 1:ctrl_g.state_dim + 1 + ctrl_g.state_dim]
            coll_aug = augmented_data[k][ctrl_g.state_dim + 1 + ctrl_g.state_dim:ctrl_g.state_dim + 1 + ctrl_g.state_dim + 1]
            r_eff_aug = augmented_data[k][ctrl_g.state_dim + 1 + ctrl_g.state_dim + 1:]
            reward_aug = (r_eff_aug + coll_aug).detach().cpu().numpy().astype(float)
            state_aug = state_aug.numpy()
            action_aug = action_aug.detach().cpu().numpy().astype(int)
            next_state_aug = next_state_aug.numpy()
            coll_aug = coll_aug.detach().cpu().numpy().astype(float)
            r_eff_aug = r_eff_aug.detach().cpu().numpy().astype(float)

           
            state_vis = state_aug[0]
            state_vis = state_vis[:,1:].reshape(60, 60)
            next_state_vis = next_state_aug[0]
            next_state_vis = next_state_vis[:,1:].reshape(60, 60)
        
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(state_vis, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Norm Speed')
            plt.title(f'Collision State')
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.subplot(1, 3, 2)
            plt.imshow(next_state_vis, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Norm Speed')
            plt.title(f'CF Collision Next State')
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.subplot(1, 3, 3)
            reward_map = np.zeros((60, 60))
            reward_map[:30, :] = coll_aug[0]
            reward_map[30:, :] = r_eff_aug[0]
            plt.imshow(reward_map, cmap='coolwarm', interpolation='nearest', vmin=-2, vmax=2)
            plt.colorbar(label='CF Norm Reward')
            plt.title(f'Safe and Effeciency')
       
          
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path, f'cf_action_{action_aug[0]}_sample_{k}.pdf'))
            plt.close()

            
            # cf_buffer.add(np.reshape(np.array([state_aug, action_aug, reward_aug, next_state_aug, False, coll_aug, r_eff_aug], dtype=object), [1, 7]))

    print("Total Reward---------------", rAll)
    if i % 100 == 0:
        torch.save({
            'model_state_dict': mainQN.state_dict(),
            'target_state_dict': targetQN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(path, f'model-{i}.pth'))
        print("Saved Model")
