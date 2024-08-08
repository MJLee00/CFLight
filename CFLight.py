
import glob
import logging
import shutil
import subprocess
import time

import numpy
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import matplotlib.pyplot as plt
import os
import scipy.stats as ss
import math
import pandas as pd
import math
import os, sys

from sumolib import checkBinary
import traci
import traci.constants as tc
import xml.etree.cElementTree as ET
import numpy as np
from buffer import priorized_experience_buffer
from model import Qnetwork
import datetime
from CTC import *
from SCM import *

net_type = 0

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
path = "./dqn"  # The path to save our model to.
h_size = 64  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001  # Rate to update target network toward primary network
DEFAULT_PORT = 8813
safe_weight = 5

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




def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

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

def state():
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
    p = traci.vehicle.getAllContextSubscriptionResults()
    p_state = np.zeros((60, 60, 2))
    for x in p:
        # TODO: 
        # 改变路网需要改变p_state
        if net_type == 0:
            ps = p[x][tc.VAR_POSITION]
            if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                continue
            spd = p[x][tc.VAR_SPEED]
            p_state[int((ps[0] - 11700)/ 5), int((ps[1]-13200) / 5)] = [1, int(round(spd)/20)]
        elif net_type == 1:
            ps = p[x][tc.VAR_POSITION]
            spd = p[x][tc.VAR_SPEED]
            p_state[int((ps[0])/ 5), int((ps[1]) / 5)] = [1, int(round(spd)/20)]
    p_state = np.reshape(p_state, [-1, 3600, 2])

   
    return p_state 


def action(round_count, tls, act, wait_time):  # parameters: the phase duration in the green signals
    tls_id = tls[0]
    init_p = act * 2
    prev = -1
    changed = False
    collision_phase = -1
    collisions = 0
    p_state = np.zeros((60, 60, 2))
    step = 0
    phase_step_counter = 0
    conflict_flag = False
    green_duration = 10
    yellow_duration = 3
    traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p].state)
    traci.trafficlight.setPhaseDuration(tls_id, green_duration)

    while traci.simulation.getMinExpectedNumber() > 0:
        if step == green_duration * 2:
            traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[
                init_p + 1].state)
            traci.trafficlight.setPhaseDuration(tls_id, yellow_duration)
        if step > green_duration * 2 + yellow_duration * 2:
            break
        traci.simulationStep()
        phase_step_counter += 1
        collisions += traci.simulation.getCollidingVehiclesNumber()
        step += 1
        if step % 10 == 0:
            for veh_id in traci.vehicle.getIDList():
                # TODO: 
                # 改变路网需要改变p_state
                if net_type == 0:
                    ps = traci.vehicle.getPosition(veh_id)
                    if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                        continue
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                elif net_type == 1:
                    ps = traci.vehicle.getPosition(veh_id)
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        
        wait_temp = dict(wait_time_map)
        for veh_id in traci.vehicle.getIDList():
            ps = traci.vehicle.getPosition(veh_id)
            spd = traci.vehicle.getSpeed(veh_id)/20
            # TODO: 
            # 改变路网需要改变p_state
            if net_type == 0:
               
                if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                    continue
                p_state[int((ps[0] - 11700)/ 5), int((ps[1]-13200) / 5)] = [1, int(round(spd))]
            elif net_type == 1:
                p_state[int((ps[0])/ 5), int((ps[1]) / 5)] = [1, int(round(spd))]

        wait_t = sum(wait_temp[x] for x in wait_temp)

        d = False
        if traci.simulation.getMinExpectedNumber() == 0:
            d = True

        r = (wait_time - wait_t)/500 - collisions/safe_weight 
        p_state_tmp = np.reshape(p_state, [-1, 3600, 2])

        
    return p_state_tmp, r, d, wait_t, collision_phase, collisions/safe_weight,  (wait_time - wait_t)/500



tf.reset_default_graph()
# define the main QN and target QN
mainQN = Qnetwork(h_size, np.int32(action_num))
targetQN = Qnetwork(h_size, np.int32(action_num))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

# define the memory
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
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)


sess = tf.InteractiveSession()

# record the loss
tf.summary.scalar('Loss', mainQN.loss)
data = []
rfile = open(path + '/reward-rl.csv', 'w')
wfile = open(path + '/wait-rl.csv', 'w')
awfile = open(path + '/acc-wait-rl.csv', 'w')
tfile = open(path + '/throput-rl.csv', 'w')

merged = tf.summary.merge_all()
s_writer = tf.summary.FileWriter(path + '/train', sess.graph)
s_writer.add_graph(sess.graph)

sess.run(init)
tf.global_variables_initializer().run()
if load_model == True:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess, ckpt.model_checkpoint_path)
updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.


gan, discriminator,generator = create_gan()

gan_ef, discriminator_ef,generator_ef = create_gan(1)

gan_coli, discriminator_coli,generator_coli = create_gan(2)

for i in range(1, num_episodes):
    episodeBuffer0 = priorized_experience_buffer()
    tls = reset()
    s = state()  
    wait_time_map = {}
    wait_time = 0 
    d = False
    rAll = 0
    j = 0
    collision_count = 0
    while j < max_epLength:
        j += 1
        if (np.random.rand(1) < e or total_steps < pre_train_steps) and not load_model:
            a = np.random.randint(0, action_num)
        else:
            np.reshape(s, [-1, 3600, 2])
            a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: s})[0]

        s1, r, d, wait_time, collision_ph, collisions, r_eff = action(i, tls, a, wait_time)

        if i > 50:
            if collisions > 0:
                collect_collision_info_plus((s,a,r,s1,collisions))
                result_list = collect_counterfactual_collision_info(generator, generator_ef, generator_coli)   
                if len(result_list) != 0:
                    for obj in result_list:
                        # 找到前一个state的索引，替换
                        episodeBuffer0.add(np.reshape(np.array([obj['state'], obj['action'],
                                    obj['reward'], obj['next_state'], obj['is_state_terminal'], obj['r_ef'], obj['colli'] ]),
                                                [1, 7]))  
                clear_collision_info_plus()
                                            
        collision_count += collisions
        total_steps += 1
        episodeBuffer0.add(np.reshape(np.array([s, a, r, s1, d, collisions, r_eff]),
                                      [1, 7]))  # Save the experience to our episode buffer.
       
        if total_steps > pre_train_steps:
            #myBuffer0.add(episodeBuffer0.buffer)
            trainBatch = myBuffer0.priorized_sample(batch_size)  # Get a random batch of experiences.
            gan_batch = np.vstack(trainBatch[:, 0])
            train_gan([np.vstack(gan_batch[:, 0]), np.eye(20)[np.vstack(gan_batch[:,1])].squeeze(),
                        np.random.normal(0, 1, (batch_size, 3600, 2))], np.vstack(gan_batch[:, 3]), gan, discriminator, generator)

            train_gan([np.vstack(gan_batch[:, 0]), np.eye(20)[np.vstack(gan_batch[:,1])].squeeze(),
                        np.random.normal(0, 1, (batch_size, 3600, 2))], np.vstack(gan_batch[:, 6]), gan_ef, discriminator_ef, generator_ef)

            train_gan([np.vstack(gan_batch[:, 0]), np.eye(20)[np.vstack(gan_batch[:,1])].squeeze(),
                        np.random.normal(0, 1, (batch_size, 3600, 2))], np.vstack(gan_batch[:, 5]), gan_coli, discriminator_coli, generator_coli)           

            if e > endE:
                e -= stepDrop
            if total_steps % (update_freq) == 0:
                
                indx = np.reshape(np.vstack(trainBatch[:, 1]), [batch_size])
                indx = indx.astype(int)
                trainBatch = np.vstack(trainBatch[:, 0])

                # Below we perform the Double-DQN update to the target Q-values
                # action from the main QN
                Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                # Q value from the target QN
                Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                # get targetQ at s'
                end_multiplier = -(trainBatch[:, 4] - 1)  # if end, 0; otherwise 1
                doubleQ = Q2[range(batch_size), Q1]
                targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)

                # Update the network with our target values.
                summry, err, ls, md = sess.run([merged, mainQN.td_error, mainQN.loss, mainQN.updateModel],
                                               feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                          mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})

                s_writer.add_summary(summry, total_steps)
                # update the target QN and the memory's prioritization
                updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
                myBuffer0.updateErr(indx, err)

        rAll += r
        s = s1

        if d == True:
            break
    # end()


    jList.append(j)
    rList.append(rAll)
    rfile.write(str(rAll) + '\n')
    wt = sum(wait_time_map[x] for x in wait_time_map)
    wtAve = wt / len(wait_time_map)
    wList.append(wtAve)
    wfile.write(str(wtAve) + '\n')
    awList.append(wt)
    awfile.write(str(wt) + '\n')
    tList.append(len(wait_time_map))
    tfile.write(str(len(wait_time_map)) + '\n')
    tmp = [x for x in wait_time_map if wait_time_map[x] > 1]
    nList.append(len(tmp) / len(wait_time_map))

    log = {'collisions': (collision_count*safe_weight)/2,
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
        df.to_csv('train_log_acyclic_R_500_sync.csv')
    elif net_type == 0:
        df.to_csv('train_log_acyclic_R_500_safeweight5.csv')
    
    end()
    myBuffer0.add(episodeBuffer0.buffer)

    

    print
    "Total Reward---------------", rAll
    # Periodically save the model.
    if i % 100 == 0:
        saver.save(sess, path + '/model-' + str(i) + '.cptk')
        print("Saved Model")
#         if len(rList) % 10 == 0:
#             print(total_steps,np.mean(rList[-10:]), e)
saver.save(sess, path + '/model-' + str(i) + '.cptk')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
