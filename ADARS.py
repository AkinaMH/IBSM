import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from Sampling import *
from Classifier import *
from Read import *
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)


# 这个方法将输入进来的 数据特征（X）和数据标签（y）合并,然后找到多数类和少数类样本并返回
def split(X, y):
    data = pd.concat((X, y), axis=1)

    maj = np.array(data[data['Class'] == 0].drop(['Class'], axis=1))
    min = np.array(data[data['Class'] == 1].drop(['Class'], axis=1))

    return maj, min


# 去除重复样本，去掉big数据集中出现的small数据样本
def duplicate_removal(big, small):
    syn = []
    for item in big.tolist():
        if item not in small.tolist():
            syn.append(item)

    syn = np.array(syn)
    return syn


datas = pima
# M是对于总样本求近邻时用到，因为ADASYN需要求附近的多数类样本
M = 10
# K这是对于样本插值生成时用到的近邻
K = 5

X = datas.iloc[:, datas.columns != "Class"]
y = datas.iloc[:, datas.columns == "Class"]

# 划分训练集（训练集：测试集=7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train = pd.concat([X_train, y_train], axis=1)

# {P}: 少数类  {N}: 多数类
P = train[train['Class'] == 1]
N = train[train['Class'] == 0]

P = P.drop(['Class'], axis=1)
N = N.drop(['Class'], axis=1)
train = train.drop(['Class'], axis=1)

train_array = train.values.tolist()
N_array = N.values.tolist()
P_array = P.values.tolist()

# {pnum}: 少数类数量  {nnum}: 多数类数量
pnum = len(P_array)
nnum = len(N_array)

DANGER = []
SAFE = []
count = 0
ri = []

# 对于整个train中求M个近邻
neigh = NearestNeighbors(n_neighbors=M + 1)
neigh.fit(train)

# step 1
for i in range(pnum):
    M_ = 0
    nnarray = neigh.kneighbors([P_array[i]], M + 1, return_distance=False)
    # nnarray_M = np.delete(nnarray[0], 0)

    for item in nnarray[0]:
        if train_array[item] in N_array:
            M_ += 1
    # 这里ri是ADASYN步骤中的参数
    ri.append(M_ / M)

    if M > M_ >= (M / 2):
        DANGER.append(i)
    elif 0 <= M_ < M / 2:
        SAFE.append(i)
    else:
        count += 1

# print(f"ri = {ri}")
# print(f"DANGER = {DANGER}")
# print(f"len(DANGER) = {len(DANGER)}")
# print(f"SAFE = {SAFE}")
# print(f"len(SAFE) = {len(SAFE)}")
# print(f"count = {count}")
# print(f"pnum = {pnum}")
# print(f"P_array[DANGER[0]] = {P_array[DANGER[0]]}")
# print(f"P_array[SAFE[0]] = {P_array[SAFE[0]]}")

# 这里ADASYN中的步骤参数
ri_hat = []
for item in ri:
    ri_hat.append(item / sum(ri))

# print('ri_hat sum = ', sum(ri_hat))

# 这里ADASYN中的步骤参数
# d  计算每个少数类需要生成的数量gi = ri_hat * G
gi = []
G = int(round((nnum - pnum) * 1))

for i in range(len(ri_hat)):
    gi.append(round(ri_hat[i] * G))

# print('gi', gi)
# print('gi len', len(gi))
# print('gi sum = ', sum(gi))

# 在这里进行ADASYN方法的样本生成，这里并不使用Python的库方法，是自己实现的了，生成过程一直到140行
# e  generate
Synthetic = []
nnarray_minority = []

# 对minority求近邻
neigh = NearestNeighbors(n_neighbors=K)
neigh.fit(P_array)

for i in range(pnum):
    nnarray_minority = neigh.kneighbors([P_array[i]], K, return_distance=False)
    # print('nnarray_minority', nnarray_minority)

    for item in range(gi[i]):
        nn = np.random.randint(0, K)
        dif = np.array(P_array[nnarray_minority[0][nn]]) - np.array(P_array[i])
        synthetic = P_array[i] + dif * np.random.rand()
        Synthetic.append(synthetic)

Synthetic = np.array(Synthetic)
# print(Synthetic)
# print(len(Synthetic))

# 找出点
# DANGER_array = []

# for item in range(len(DANGER)):
#     # print(item)
#     DANGER_array.append(P_array[DANGER[item]])

# print(f"DANGER_array = {DANGER_array}")
# print(f"len(DANGER_array) = {len(DANGER_array)}")

# 找到需要使用RSMOTE方法的点，用SAFE表示
SAFE_array = []

for item in range(len(SAFE)):
    # print(item)
    SAFE_array.append(P_array[SAFE[item]])

# print(f"SAFE_array = {SAFE_array}")
# print(f"len(SAFE_array) = {len(SAFE_array)}")

# DANGER_frame = pd.DataFrame(DANGER_array)
# DANGER_frame['Class'] = 1
# print(DANGER_frame)

SAFE_frame = pd.DataFrame(SAFE_array)
SAFE_frame['Class'] = 1
# print(SAFE_frame)

N['Class'] = 0
# print(N)
N.columns = [0, 1, 2, 3, 4, 5, 6, 7, 'Class']

# ada_frame = pd.concat([N, DANGER_frame], axis=0)
# print(ada_frame)
# 将多数类 N 与 safe区域的少数类合并起来
rs_frame = pd.concat([N, SAFE_frame], axis=0)
# print(rs_frame)

# 区分出 样本特征（X_rs）和样本标签（y_rs）
X_rs = rs_frame.iloc[:, rs_frame.columns != "Class"]
y_rs = rs_frame.iloc[:, rs_frame.columns == "Class"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# X_adasyn, y_adasyn = adasyn(X_train, y_train)
# maj_adasyn, min_adasyn = split(X_adasyn, y_adasyn)
# print(X_adasyn)
# print(y_adasyn)
# syn_adasyn = duplicate_removal(min_adasyn, np.array(P))
# print(syn_adasyn)

# 这里调用RSMOTE进行safe区域样本的生成
X_train_rsmote = np.array(X_rs)
y_train_rsmote = np.array(y_rs).flatten()
X_rsmote, y_rsmote = rsmote(X_train_rsmote, y_train_rsmote)

maj_rsmote, min_rsmote = split(pd.DataFrame(X_rsmote), pd.DataFrame(y_rsmote, columns=['Class']))
# print(X_rsmote)
# print(y_rsmote)
syn_rsmote = duplicate_removal(min_rsmote, np.array(P_array))
# print(syn_rsmote)
# print(len(syn_rsmote))
# print(type(syn_rsmote))

# 将adasyn方法生成的样本（Synthetic）与rsmote方法生成的样本（syn_rsmote） 两个合成样本拼接
syn = np.concatenate((Synthetic, syn_rsmote), axis=0)
# print(syn)
# print(len(syn))
# print(type(syn))

# 以下两行转换成DataFrame格式方便后续使用，syn_frame['Class'] = 1给生成样本赋值为1（即代表该样本是少数类样本）
syn_frame = pd.DataFrame(syn)
syn_frame['Class'] = 1
# print(syn_frame)
# print(N)
# 将合成的少数类样本和原先的多数类样本合并，得到最终总的数据集
final_frame = pd.concat([N, syn_frame], axis=0)

# 区分特征的标签
X_syn = final_frame.iloc[:, final_frame.columns != "Class"]
y_syn = final_frame.iloc[:, final_frame.columns == "Class"]

# 因为这里首先将样本进行划分了训练集和测试集，所以直接进行分类器测试
adars_dt_scores = DecisionTree("ADARS", X_syn, X_test, y_syn, y_test)
adars_rf_scores = RandomForest("ADARS", X_syn, X_test, y_syn, y_test)
adars_svm_scores = SVM("ADARS", X_syn, X_test, y_syn, y_test)
adars_nb_scores = NB("ADARS", X_syn, X_test, y_syn, y_test)
adars_knn_scores = KNN("ADARS", X_syn, X_test, y_syn, y_test)
adars_lr_scores = LR("ADARS", X_syn, X_test, y_syn, y_test)
adars_ada_scores = Adaboost("ADARS", X_syn, X_test, y_syn, y_test)

