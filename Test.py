import math
import random
from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import *
from sklearn.metrics import SCORERS
from Read import *

print("随机选择几个元素")
array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print('array = ', array)

select = random.sample(array, 5)
print('select = ', select)

s = np.random.randint(1, 5)
print(s)

s_nn = random.sample(array, s)
print(s_nn)

print("=" * 40)
print("向下取整")

num1 = 3.4
num2 = 3.5
num3 = 3.6

print("num1 floor", math.floor(num1))
print("num2 floor", math.floor(num2))
print("num3 floor", math.floor(num3))

print("=" * 40)
print("删除numpy数组第一个元素")
a = np.array([1, 2, 3, 4, 5, 6])
print(a)
aa = np.delete(a, 1)
print(aa)

print("=" * 40)
print("计算欧氏距离")

a = [7.0, 184.0, 84.0, 33.0, 0.0, 35.5, 0.355, 41.0]
b = [3.0, 174.0, 58.0, 22.0, 194.0, 32.9, 0.593, 36.0]
c = [5.0, 168.0, 64.0, 0.0, 0.0, 32.9, 0.135, 41.0]

b_a = np.linalg.norm(np.array(b) - np.array(a))
c_a = np.linalg.norm(np.array(c) - np.array(a))

print(b_a)
print(c_a)

print("=" * 40)
print("找最小下标")

list = [-1, 3, -5, 7, 10, 9]

print(list.index(min(list)))
print(list)
list.pop(list.index(min(list)))
print(list)

list.remove(10)
print(list)

print("=" * 40)
print("可以使用的评分方法")
print(SCORERS.keys())

print("=" * 40)
print("numpy")

a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
b = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
scores = np.array([])
scores = np.append(scores, 0.2)
scores = np.append(scores, 0.3)
scores = np.append(scores, 0.6)
print(scores)

print(a + b)
print((a + b) / 2)

a = np.array([0.74675325, 0.72727273, 0.625, 0.67226891, 0.74242424])
b = np.array([0.72077922, 0.57446809, 0.54, 0.55670103, 0.67975741])
c = np.array([0.72727273, 0.59016393, 0.67924528, 0.63157895, 0.70368412])
d = np.array([0.71895425, 0.57446809, 0.54, 0.55670103, 0.67874348])
e = np.array([0.7254902, 0.60344828, 0.64814815, 0.625, 0.70172414])
print(a + b + c + d + e)
print((a + b + c + d + e) / 5)

aa = np.array([0.72077922, 0.6, 0.61111111, 0.60550459, 0.80367309])
bb = np.array([0.78571429, 0.65957447, 0.64583333, 0.65263158, 0.81745874])
cc = np.array([0.79220779, 0.57377049, 0.85365854, 0.68627451, 0.82416711])
dd = np.array([0.76470588, 0.55319149, 0.63414634, 0.59090909, 0.8272782])
ee = np.array([0.75816993, 0.5862069, 0.72340426, 0.64761905, 0.85245009])
print(aa + bb + cc + dd + ee)
print((aa + bb + cc + dd + ee) / 5)

print("=" * 40)
print("read 5-fold")
result = []
file = yeast3_filenames
for i in range(len(file))[::2]:
    if i + 1 < len(file):
        result.append((file[i], file[i + 1]))
    else:
        result.append((file[i],))
    print(result)
    print(result[0][0])
    print(result[0][1])
    train = pd.read_csv(result[0][0])
    test = pd.read_csv(result[0][1])
    # print(train)
    # print(test)

    X_train = train.iloc[:, train.columns != "Class"]
    y_train = train.iloc[:, train.columns == "Class"]
    X_test = test.iloc[:, test.columns != "Class"]
    y_test = test.iloc[:, test.columns == "Class"]

    print('X_train', X_train)
    print('y_train', y_train)
    print('X_test', X_test)
    print('y_test', y_test)

    # X = data.iloc[:, data.columns != "Class"]
    # y = data.iloc[:, data.columns == 'Class']
    result = []
    print("=" * 20)

# l = [1, 2, 3, 4, 5, 6]
# result = []
# for i in range(len(l))[::2]:
#     if i + 1 < len(l):
#         result.append((l[i], l[i + 1]))
#     else:
#         result.append((l[i],))
#     print(result)
#     print(result[0][0])
#     print(result[0][1])
#     result = []

print("=" * 40)
print("")

x = [57.07168, 46.95301, 31.86423, 38.27486, 77.89309, 76.78879, 33.29809, 58.61569, 18.26473, 62.92256, 50.46951,
     19.14473, 22.58552, 24.14309]
y = [8.319966, 2.569211, 1.306941, 8.450002, 1.624244, 1.887139, 1.376355, 2.521150, 5.940253, 1.458392, 3.257468,
     1.574528, 2.338976]
print(ranksums(x, y))

x = ['d', 'd', 'a', 'b', 'c']
y = ['d', 'a']
z = []
for m in x:
    if m not in y:
        z.append(m)

print(z)
