import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Protected-only', 'Permitted-only']
wait_times = [117.01, 5.54]  # 平均等待时间 (秒)
collisions = [0.8, 40.04]    # 碰撞次数

# 设置柱状图的参数
x = np.arange(len(categories))  # 横轴位置
width = 0.35  # 柱子宽度

# 创建图形
fig, ax1 = plt.subplots(figsize=(6, 4))

# 绘制第一个柱状图：平均等待时间（左y轴）
bars1 = ax1.bar(x - width/2, wait_times, width, label='Average wait time (s)', color='red')
ax1.set_ylabel('Average wait time (s)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# 创建第二个y轴，绘制碰撞次数（右y轴）
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, collisions, width, label='Collision', color='blue')
ax2.set_ylabel('Collision', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# 设置x轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(categories)

# 添加数据标签
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}', ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}', ha='center', va='bottom')

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

# 设置标题
plt.title('Performance comparison')

# 调整布局
plt.tight_layout()

# 保存为 PDF
plt.savefig('compare.svg', format='svg', bbox_inches='tight')

# 显示图形
plt.show()