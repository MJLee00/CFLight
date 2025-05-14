import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# # 定义目录和标签变量
draw_dir = 'R-draw'  # 可以改为 'Q-draw' 或 'Loss-draw'
output_name = 'compare_r_real'  # 输出文件名前缀
Base_1 = 'CFLight-R'
Base_2 = 'Sync-R'


# 定义目录和标签变量
# draw_dir = 'Q-draw'  # 可以改为 'Q-draw' 或 'Loss-draw'
# output_name = 'compare_q_real'  # 输出文件名前缀
# Base_1 = 'CFLight-Q'
# Base_2 = 'Sync-Q'


# draw_dir = 'Loss-draw'  # 可以改为 'Q-draw' 或 'Loss-draw'
# output_name = 'compare_loss_real'  # 输出文件名前缀
# Base_1 = 'CFLight-Loss'
# Base_2 = 'Safe-Loss'


# 读取所有CSV文件
data_dict = {}
for file in os.listdir(draw_dir):
    if file.endswith('.csv'):
        name = file.replace('.csv', '')
        data_dict[name] = pd.read_csv(os.path.join(draw_dir, file))

# 定义颜色方案
colors = {
    Base_1: 'red',  # 突出显示CFLight-R
    'Safe-Act': 'blue',
    Base_2: 'green',
    '3DQN': 'gray'
}

# 定义平滑函数
def smooth_data(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 创建图形
plt.figure(figsize=(15, 10))

# 第一个子图 - collisions
plt.subplot(2, 1, 1)
for name, data in data_dict.items():
    # 原始数据
    plt.plot(range(len(data)), data['collisions'], color=colors[name], alpha=0.1)
    # 平滑后的数据
    plt.plot(range(len(smooth_data(data['collisions']))), 
             smooth_data(data['collisions']), 
             label=name, color=colors[name], 
             linewidth=2 if name == 'CFLight-R' else 1)

plt.xlabel('Epoch')
plt.ylabel('Collisions')
plt.title('Collisions Comparison')
plt.legend()
plt.grid(True)
# 调整x轴和y轴的刻度位置
plt.gca().spines['left'].set_position(('data', 0))
plt.gca().spines['bottom'].set_position(('data', 0))

# 第二个子图 - wait_average
plt.subplot(2, 1, 2)
for name, data in data_dict.items():
    # 原始数据
    plt.plot(range(len(data)), data['wait_average'], color=colors[name], alpha=0.1)
    # 平滑后的数据
    plt.plot(range(len(smooth_data(data['wait_average']))), 
             smooth_data(data['wait_average']), 
             label=name, color=colors[name], 
             linewidth=2 if name == 'CFLight-R' else 1)

plt.xlabel('Epoch')
plt.ylabel('Wait Average')
plt.title('Wait Average Comparison')
plt.legend()
plt.grid(True)
# 调整x轴和y轴的刻度位置
plt.gca().spines['left'].set_position(('data', 0))
plt.gca().spines['bottom'].set_position(('data', 0))

# 调整子图之间的间距
plt.tight_layout()
plt.savefig(f'results/{output_name}.pdf')
#