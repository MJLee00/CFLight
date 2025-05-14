import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
# Monotonic MLP for SCM
class MonotonicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MonotonicMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current_dim = input_dim
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        self.layers.append(nn.Linear(current_dim, output_dim))
        # He initialization for ReLU
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        self.monotonicity_penalty = 1.0 

    def forward(self, x):
        input_x = x
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            residual = x if x.shape[-1] == layer.out_features else layer(input_x)  # Residual connection
            x = layer(x)
            x = norm(x)
            x = torch.relu(x) + residual
        x = self.layers[-1](x)
        return x

    def get_monotonicity_penalty(self):
        penalty = 0
        for layer in self.layers:
            penalty += torch.sum(torch.relu(-layer.weight)) * self.monotonicity_penalty
        return penalty


# BiCoGAN Components
class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, reward_dim):
        super(Generator, self).__init__()
        self.state_model = MonotonicMLP(state_dim + action_dim + noise_dim_s, hidden_dim, state_dim)
        self.reward_model = MonotonicMLP(state_dim + action_dim + noise_dim_r, hidden_dim, reward_dim)

    def forward(self, state, action, noise_s, noise_r):
        state_input = torch.cat([state, action, noise_s], dim=-1)
        next_state = torch.sigmoid(self.state_model(state_input))
        reward_input = torch.cat([state, action, noise_r], dim=-1)
        reward = self.reward_model(reward_input)
        return next_state, reward

    def get_monotonicity_penalty(self):
        return self.state_model.get_monotonicity_penalty() + self.reward_model.get_monotonicity_penalty()


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, reward_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + reward_dim, hidden_dim),  # Input: (S_{t+1}, R_t)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + action_dim + noise_dim_s + noise_dim_r)
        )
        # 添加噪声生成网络
        self.noise_s_model = nn.Sequential(
            nn.Linear(state_dim + reward_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, noise_dim_s),
            nn.Tanh()  # 使用Tanh将输出限制在[-1,1]
        )
        self.noise_r_model = nn.Sequential(
            nn.Linear(state_dim + reward_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, noise_dim_r),
            nn.Tanh()  # 使用Tanh将输出限制在[-1,1]
        )

    def forward(self, next_state, reward):
        x = torch.cat([next_state, reward], dim=-1)
        features = self.model(x)
        # 生成并限制噪声
        noise_s = self.noise_s_model(x)
        noise_r = self.noise_r_model(x)
        return features, noise_s, noise_r

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, reward_dim, num_layers=3):
        super(Discriminator, self).__init__()
        input_dim = 2 * state_dim + action_dim + reward_dim + noise_dim_s + noise_dim_r
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())   
            self.layers.append(nn.Dropout(0.3))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers.append(nn.Sigmoid())
  
    def forward(self, state, action, next_state, reward, noise_s, noise_r):
        x = torch.cat([state, action, next_state, reward, noise_s, noise_r], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x

# D3QN
class D3QN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(D3QN, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, 2)  # CartPole只有2个动作

    def forward(self, state):
        x = self.shared(state)
        value = self.value(x)
        advantage = self.advantage(x)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

# CTRL_g Algorithm
class CTRLG:
    def __init__(self, state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, reward_dim, batch_size=64, lr_g=1e-3, lr_d=1e-4):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.generator = Generator(state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, reward_dim).to(self.device)
        self.encoder = Encoder(state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, reward_dim).to(self.device)
        self.discriminator = Discriminator(state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, reward_dim).to(self.device)
        self.d3qn = D3QN(state_dim, 2, hidden_dim).to(self.device)  # 固定为2个动作
        self.g_optimizer = optim.AdamW(self.generator.parameters(), lr=lr_g, weight_decay=1e-5)
        self.e_optimizer = optim.AdamW(self.encoder.parameters(), lr=lr_g, weight_decay=1e-5)
        self.d_optimizer = optim.AdamW(self.discriminator.parameters(), lr=lr_d, weight_decay=1e-5)
        self.g_scheduler = CosineAnnealingLR(self.g_optimizer, T_max=100)
        self.e_scheduler = CosineAnnealingLR(self.e_optimizer, T_max=100)
        self.d_scheduler = CosineAnnealingLR(self.d_optimizer, T_max=100)
        self.d3qn_optimizer = optim.Adam(self.d3qn.parameters(), lr=1e-5)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noise_dim_s = noise_dim_s
        self.noise_dim_r = noise_dim_r
        self.reward_dim = reward_dim

    def train_bicogan(self, data, epochs=100, batch_size=64, label_smoothing=True):
        for epoch in range(epochs):
            indices = torch.randperm(data.shape[0])
            d_losses, g_losses, e_losses = [], [], []
            for start_idx in range(0, data.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, data.shape[0])
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                
                states = batch_data[:, :self.state_dim]
                actions = batch_data[:, self.state_dim:self.state_dim+1].long()  # 改为float类型
                actions = torch.nn.functional.one_hot(actions.squeeze(), self.action_dim).float()
                next_states = batch_data[:, self.state_dim+1:self.state_dim+1+self.state_dim]
                rewards = batch_data[:, -self.reward_dim:]
                
                real_label = torch.full((states.shape[0], 1), 0.9 if label_smoothing else 1.0).to(self.device)
                fake_label = torch.full((states.shape[0], 1), 0.1 if label_smoothing else 0.0).to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                inferred, inferred_noise_s, inferred_noise_r = self.encoder(next_states, rewards)
                inferred_states = inferred[:, :self.state_dim]
                inferred_actions = inferred[:, self.state_dim:self.state_dim+self.action_dim]
      
                real_output = self.discriminator(states, actions, next_states, rewards, inferred_noise_s, inferred_noise_r)
                d_loss_real = self.bce_loss(real_output, real_label)
                
                noise_s = torch.randn(states.shape[0], self.noise_dim_s).to(self.device)
                noise_r = torch.randn(states.shape[0], self.noise_dim_r).to(self.device)
                generated_next_states, generated_rewards = self.generator(states, actions, noise_s, noise_r)
                fake_output = self.discriminator(states, actions, generated_next_states, generated_rewards, noise_s, noise_r)
                d_loss_fake = self.bce_loss(fake_output, fake_label)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                

                for _ in range(5):
                    # Train Generator and Encoder
                    self.g_optimizer.zero_grad()
                    self.e_optimizer.zero_grad()
                    inferred, inferred_noise_s, inferred_noise_r = self.encoder(next_states, rewards)
                    inferred_states = inferred[:, :self.state_dim]
                    inferred_actions = inferred[:, self.state_dim:self.state_dim+self.action_dim]
        
                    generated_next_states, generated_rewards = self.generator(states, actions, inferred_noise_s, inferred_noise_r)
                    fake_output = self.discriminator(states, actions, generated_next_states, generated_rewards, inferred_noise_s, inferred_noise_r)
                    g_loss = self.bce_loss(fake_output, real_label)
                    r_loss = self.mse_loss(generated_rewards, rewards)
                    state_loss = self.mse_loss(generated_next_states, next_states)
                    e_loss = self.mse_loss(inferred_states, states) + nn.CrossEntropyLoss()(inferred_actions, actions)
                    + self.generator.get_monotonicity_penalty()
                    (g_loss + e_loss + r_loss + state_loss).backward()
                    self.g_optimizer.step()
                    self.e_optimizer.step()
            self.g_scheduler.step()
            self.e_scheduler.step()
            self.d_scheduler.step()    
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, E Loss: {e_loss.item():.4f}")

    def generate_counterfactuals(self, data, coll, action_space=10):
        augmented_data = []
        states = data[:, :self.state_dim]
        actions = data[:, self.state_dim:self.state_dim+1].long()  # 改为float类型
        coll = torch.tensor(coll, dtype=torch.float32).to(self.device)
        next_states = data[:, self.state_dim+1:-self.reward_dim]
        rewards = data[:, -self.reward_dim:]
        
        with torch.no_grad():   
            inferred, inferred_noise_s, inferred_noise_r = self.encoder(next_states, rewards)
            noise_s = inferred_noise_s
            noise_r = inferred_noise_r
        
        
        for i in range(action_space):
            # 获取不等于3,4,8,9的索引
            if i == actions.item():
                # 添加到增强数据中
                augmented_data.append(torch.cat([
                    states, 
                    actions, 
                    next_states, 
                    rewards
                ], dim=-1))
                continue
            else:

                new_actions = actions.clone()

                for j in range(len(new_actions)):
                    action = new_actions[j].item()
                    new_actions[j] = i
                
                # 转换为one-hot编码
                new_actions_one_hot = torch.nn.functional.one_hot(new_actions.squeeze(), self.action_dim).float()
                
                # 生成新的状态和奖励
                with torch.no_grad():
                    new_next_states, new_rewards = self.generator(
                        states, 
                        new_actions_one_hot.view(1, -1), 
                        noise_s, 
                        noise_r
                    )
                
                # 调整奖励
                new_rewards[:, -2] = (new_rewards[:, -2])
                
                # 添加到增强数据中
                augmented_data.append(torch.cat([
                    states, 
                    new_actions, 
                    new_next_states, 
                    new_rewards
                ], dim=-1))
                
        return torch.cat(augmented_data, dim=0) if augmented_data else torch.tensor([]).to(self.device)

    def train_d3qn(self, data, epochs=100, gamma=0.99, batch_size=64):
        for epoch in range(epochs):
            indices = torch.randperm(data.shape[0])
            d3qn_losses = []
            
            for start_idx in range(0, data.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, data.shape[0])
                batch_indices = indices[start_idx:end_idx]
                batch_data = data[batch_indices]
                states = batch_data[:, :self.state_dim]
                actions = batch_data[:, self.state_dim:self.state_dim+self.action_dim].long()  # 确保动作是长整型
                next_states = batch_data[:, self.state_dim+self.action_dim:-1]
                rewards = batch_data[:, -1:]
                
                self.d3qn_optimizer.zero_grad()
                
                # 获取当前状态的Q值
                q_values = self.d3qn(states)
                # 使用gather选择对应动作的Q值
                q_values = q_values.gather(1, actions)
                
                # 获取下一状态的最大Q值
                with torch.no_grad():
                    next_q_values = self.d3qn(next_states)
                    next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                
                # 计算目标Q值
                targets = rewards + gamma * next_q_values
                
                # 计算损失
                loss = self.mse_loss(q_values, targets)
                loss.backward()
                self.d3qn_optimizer.step()
                
            if epoch % 10 == 0:
                print(f"D3QN Epoch {epoch}, Loss: {loss.item():.4f}")

# Example Usage
def main():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # 动作维度为1，但实际值是0或1
    noise_dim_s = 4
    noise_dim_r = 4
    hidden_dim = 600
    
    # Create dataset (simplified SD dataset)
    data = []
    for _ in range(250):  # 250 trials
        state = env.reset()
        state = state[0]
        for _ in range(20):  # 20 steps
            action = np.random.randint(0, 2)  # 随机选择0或1
            next_state, reward, done, _, _ = env.step(action)
            data.append(np.concatenate([state, [action], next_state, [reward]]))
            state = next_state
            if done:
                break
    data = torch.tensor(data, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Train CTRL_g
    ctrl_g = CTRLG(state_dim, action_dim, noise_dim_s, noise_dim_r, hidden_dim, batch_size=64)
    ctrl_g.train_bicogan(data, epochs=100, batch_size=ctrl_g.batch_size)
    augmented_data = ctrl_g.generate_counterfactuals(data, action_space=2)
    combined_data = torch.cat([data, augmented_data], dim=0)
    #combined_data = data
    # Train D3QN
    ctrl_g.train_d3qn(combined_data, batch_size=ctrl_g.batch_size)
    
    # Evaluate
    total_reward = 0
    for _ in range(10):
        state = env.reset()
        state = state[0]
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(ctrl_g.device)
            action = ctrl_g.d3qn(state_tensor).argmax().item()  # 直接使用0或1
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
    print(f"Average Reward: {total_reward / 10}")

if __name__ == "__main__":
    main()