import traci 
import pickle 
import numpy as np
import traci.constants as tc
import os 
import shutil
import tensorflow as tf

file_dir = "./cologne1/"
start_time = 25200
back_step = 3
step_length = 9
phase_list = [0,2,4,6,8,10,12,14,16,18]
try_times = 50
collision_trajectory = []

def clear_collision_info_plus():
    collision_trajectory.clear()

def collect_collision_info_plus(batch):
    collision_trajectory.append(batch)    

def collect_counterfactual_collision_info(generator, generator_ef, generator_coli):
    result_list = []
    for s,a,r,s1,actual_collision_count in collision_trajectory:
        a = a*2
        if a == 6 or a == 16:
            break
        counter_a = 6
        if a == 0 or a == 2 or a == 4:
            counter_a = 6
        if a == 10 or a == 12 or a == 14:
            counter_a = 16
      

        # 创建一个 TensorFlow 会话
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 运行计算图以获取 model_output 的值
            next_state = sess.run(generator.output, feed_dict={generator.input[0]:s, generator.input[1]:np.eye(20)[counter_a].reshape(1,-1),
                    generator.input[2]:np.random.normal(0, 1, (1, 3600, 2))})
            r_ef = sess.run(generator_ef.output, feed_dict={generator_ef.input[0]:s, generator_ef.input[1]:np.eye(20)[counter_a].reshape(1,-1),
                    generator_ef.input[2]:np.random.normal(0, 1, (1, 3600, 2))})
            collision = sess.run(generator_coli.output, feed_dict={generator_coli.input[0]:s, generator_coli.input[1]:np.eye(20)[counter_a].reshape(1,-1),
                    generator_coli.input[2]:np.random.normal(0, 1, (1, 3600, 2))})
  
        result_list.append({'state':  s,
                            'action':  int(counter_a/2),
                            'reward': r_ef+ (actual_collision_count-collision)*500,
                            'next_state' : next_state,
                            'is_state_terminal': False,
                            'r_ef': r_ef,
                            'colli': collision})

    return result_list