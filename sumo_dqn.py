import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from sumo_env import SumoEnv

# Create the environment
env = SumoEnv()

# Create the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 定义 DQNAgent 类
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95    # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=3, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_test=False):
        if is_test is not True:
            if np.random.rand() <= self.epsilon:
                return random.randrange(2)
            act_values = self.model.predict(state)
        
            return np.argmax(act_values[0])
        else:
            act_values = self.model.predict(state)
        
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=100, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

state_size = env.observation_space
action_size = env.action_space
agent = DQNAgent(state_size, action_size)

r = []
# 训练智能体
batch_size = 32
for episode in range(200):
    re = []
    state = env.reset()
    state = np.reshape(state, [1, 3])
    total_reward = 0
    for t in range(100):
        if episode > 100:
            action = agent.act(state, True)
        else:
            action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if episode > 100:
            re.append(reward)
        next_state = np.reshape(next_state, [1, 3])
        if episode < 100:
            #反事实
            if reward == -1:
                env.close()
                state_cf = env.reset()
                for ste in range(100):
                    state_cf = np.reshape(state_cf, [1, 3])
                    next_state_cf, reward_cf, d, _ = env.step_cf(1)
                    next_state_cf = np.reshape(next_state_cf, [1, 3])
                    if ste == 10:
                        agent.remember(state_cf, 1, reward_cf, next_state_cf, False)
                        break
                   
            
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    env.close()
    r.append(re)
    

import matplotlib.pyplot as plt
import pickle

# 定义数组
array = r
pkl_file_path = "cf.pkl"

# 使用pickle模块将array保存为pkl文件
with open(pkl_file_path, "wb") as f:
    pickle.dump(array, f)

# 创建 x 坐标轴
x = range(len(array))

# 创建图形和轴对象
fig, ax = plt.subplots()

# 绘制直线
ax.plot(x, array, marker='o', linestyle='-')

# 设置图形标题和轴标签
ax.set_title('Array as Line Plot')
ax.set_xlabel('Index')
ax.set_ylabel('Value')

plt.savefig('array_plot.png')