import pickle
import csv
import pandas as pd
import numpy as np
# 指定pickle文件的路径
pkl_file_path = "array.pkl"

# 读取pickle文件中的数据
with open(pkl_file_path, "rb") as f:
    array1 = pickle.load(f)

array1 = [sum(x) for x in array1][100:]

pkl_file_path = "cf.pkl"

# 读取pickle文件中的数据
with open(pkl_file_path, "rb") as f:
    array2 = pickle.load(f)
array2 = [sum(x) for x in array2][100:]

data = array2
mean_value = np.mean(data)

# 创建 x 坐标轴
x = np.arange(len(data))
import matplotlib.pyplot as plt
# 创建图形和轴对象
fig, ax = plt.subplots()

# 绘制数据曲线
ax.plot(x, data)

# 绘制平均线
ax.axhline(y=mean_value, color='r', linestyle='--')


# 设置图形标题和轴标签
ax.set_title('Data with Mean Line')
ax.set_xlabel('Index')
ax.set_ylabel('Value')

plt.savefig('test.svg')


df = pd.DataFrame({'original': array1, 'cf+': array2})

# 指定CSV文件的路径
csv_file_path = "array.csv"

# 将DataFrame写入CSV文件
df.to_csv(csv_file_path, index=False)
