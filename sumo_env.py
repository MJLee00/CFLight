import os
import numpy as np
import traci
from gym import Env
from gym.spaces import Discrete, Box
import math 
class SumoEnv(Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        # Define action and observation space
        # Actions: 0 = accelerate, 1 = decelerate, 2 = keep constant speed
        self.action_space = Discrete(2)
        # Observations: [speed of car, distance to front car, acceleration of car]
        self.observation_space = Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.rewards=[]
        # Path to SUMO config file
        self.sumo_cfg = "/root/cflight_submit/car/simple.sumocfg"
    
    def reset(self):
        # Start a new simulation instance
        traci.start(['sumo', '-c', self.sumo_cfg])
        # Initialize environment state
        self.update_state()
        return self.state
    
    def step(self, action):
        # Perform the action
       
        reward = 0
        self.update_state()   
        if action == 0:
            traci.vehicle.changeLane("car2", 0, 1000)
        elif action == 1:
            pass
        traci.simulationStep()
            
        if traci.simulation.getCollidingVehiclesNumber() > 0:  # if collision
            reward = -1

            
     
        done = reward == -1  # Define when to finish an episode

        return self.state, reward, done, {}

    def step_cf(self, action):
        # Perform the action
       
        reward = 1
        self.update_state() 
            
        if action == 0:
            traci.vehicle.changeLane("car2", 0, 1000)
        elif action == 1:
            pass

        traci.simulationStep()
        
        done = traci.simulation.getMinExpectedNumber() == 0  # Define when to finish an episode
        if done:
            self.close()
        return self.state, reward, done, {}


    def calculate_distance(self,pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def update_state(self):
        car1_speed = traci.vehicle.getSpeed("car1")
        car2_speed = traci.vehicle.getSpeed("car2")
        pos1 = traci.vehicle.getPosition("car1")
        pos2 = traci.vehicle.getPosition("car2")
        distance_to_front_car = self.calculate_distance(pos1, pos2)

        self.state = np.array([car1_speed, distance_to_front_car, car2_speed])

    def close(self):
        traci.close()
