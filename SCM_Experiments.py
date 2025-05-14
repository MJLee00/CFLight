import torch
import numpy as np
import time
from gan_cf import CTRLG
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 初始化参数
buffer_size = 20000
state_dim = 3600
action_dim = 10
noise_dim_s = 4
noise_dim_r = 4
hidden_dim = 600
reward_dim = 2
batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化CTRLG模型
ctrl_g = CTRLG(state_dim=state_dim, 
               lr_g=1e-4, 
               lr_d=1e-3, 
               action_dim=action_dim, 
               noise_dim_s=noise_dim_s, 
               noise_dim_r=noise_dim_r, 
               hidden_dim=hidden_dim, 
               reward_dim=reward_dim, 
               batch_size=batch_size)

def generate_collision_data(collision_ratio):
    """生成碰撞数据，其中碰撞数据占比为collision_ratio"""
    n_collision = int(buffer_size * collision_ratio)
    
    # 只生成碰撞数据

    collision_states = np.random.randn(n_collision, state_dim)
    collision_actions = np.random.randint(0, action_dim, size=(n_collision, 1))
    collision_next_states = np.random.randn(n_collision, state_dim)
    collision_collisions = np.ones((n_collision, 1))
    collision_r_eff = np.random.randn(n_collision, 1)
    
    return collision_states, collision_actions, collision_next_states, collision_collisions, collision_r_eff

def test_training_performance():
    """测试不同碰撞比例下的训练性能"""
    collision_ratios = [0.1, 0.2, 0.3, 0.4]
    training_times = []
    
    for ratio in collision_ratios:
        print(f"\n测试碰撞比例 {ratio*100}% 的训练性能")
        
        # 生成数据
        states, actions, next_states, collisions, r_eff = generate_collision_data(ratio)
        
        # 准备数据
        data = np.concatenate([states, actions, next_states, collisions, r_eff], axis=1)
        data = torch.tensor(data, dtype=torch.float32).to(device)
        
        # 记录训练时间
        start_time = time.time()
        ctrl_g.train_bicogan(data, epochs=500, batch_size=batch_size)
        end_time = time.time()
        
        training_time = end_time - start_time
        training_times.append(training_time)
        print(f"训练时间: {training_time:.2f} 秒")
        print(f"训练数据量: {len(data)} 条")
    
    return collision_ratios, training_times

def test_generation_performance():
    """测试不同碰撞比例下的生成性能"""
    collision_ratios = [0.0005]
    generation_times = []
    
    for ratio in collision_ratios:
        print(f"\n测试碰撞比例 {ratio*100}% 的生成性能")
        
        # 生成数据
        states, actions, next_states, collisions, r_eff = generate_collision_data(ratio)
        
        # 准备数据
        data = np.concatenate([states, actions, next_states, collisions, r_eff], axis=1)
        data = torch.tensor(data, dtype=torch.float32).to(device)
        
        # 确保collisions是1D张量
        collisions = torch.tensor(collisions, dtype=torch.float32).to(device).flatten()
        
        # 记录生成时间
        try:
            start_time = time.time()
            augmented_data = ctrl_g.generate_counterfactuals(data, -collisions, action_space=action_dim)
            end_time = time.time()
            
            generation_time = end_time - start_time
            generation_times.append(generation_time)
            print(f"生成时间: {generation_time:.2f} 秒")
            print(f"生成数据量: {len(data)} 条")
        except RuntimeError as e:
            print(f"生成错误: {e}")
            generation_times.append(None)
    
    return collision_ratios, generation_times

def plot_results(collision_ratios, training_times, generation_times):
    """绘制结果图表"""
    plt.figure(figsize=(12, 5))
    
    # 训练时间图
    plt.subplot(1, 2, 1)
    plt.plot(collision_ratios, training_times, 'b-o')
    plt.xlabel('碰撞数据占比')
    plt.ylabel('训练时间 (秒)')
    plt.title('不同碰撞比例下的训练时间\n(仅使用碰撞数据)')
    plt.grid(True)
    
    # 生成时间图
    plt.subplot(1, 2, 2)
    valid_ratios = [r for r, t in zip(collision_ratios, generation_times) if t is not None]
    valid_times = [t for t in generation_times if t is not None]
    plt.plot(valid_ratios, valid_times, 'r-o')
    plt.xlabel('碰撞数据占比')
    plt.ylabel('生成时间 (秒)')
    plt.title('不同碰撞比例下的生成时间\n(仅使用碰撞数据)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ctrl_g_collision_performance.pdf')
    plt.close()

def main():
    print("开始性能测试...")
    print("注意：本次测试仅使用碰撞数据")
    
    # 测试训练性能
    print("\n=== 测试训练性能 ===")
    #collision_ratios, training_times = test_training_performance()
    
    # 测试生成性能
    print("\n=== 测试生成性能 ===")
    collision_ratios, generation_times = test_generation_performance()
    
    # 绘制结果
    plot_results(collision_ratios, training_times, generation_times)
    
    # 保存结果到CSV
    results = pd.DataFrame({
        'collision_ratio': collision_ratios,
        #'training_time': training_times,
        'generation_time': generation_times,
        'data_size': [int(buffer_size * ratio) for ratio in collision_ratios]
    })
    results.to_csv('ctrl_g_collision_performance_results.csv', index=False)
    
    print("\n测试完成！结果已保存到 ctrl_g_collision_performance_results.csv 和 ctrl_g_collision_performance.pdf")

if __name__ == "__main__":
    main()