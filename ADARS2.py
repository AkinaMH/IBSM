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


def split(X, y):
    data = pd.concat((X, y), axis=1)

    maj = np.array(data[data['Class'] == 0].drop(['Class'], axis=1))
    min = np.array(data[data['Class'] == 1].drop(['Class'], axis=1))

    return maj, min


def duplicate_removal(big, small):
    syn = []
    for item in big.tolist():
        if item not in small.tolist():
            syn.append(item)

    syn = np.array(syn)
    return syn

data = pima

K = 5

# X = datas.iloc[:, datas.columns != "Class"]
# y = datas.iloc[:, datas.columns == "Class"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
# data = pd.concat([X_train, y_train], axis=1)

# {P}: 少数类  {N}: 多数类
P = data[data['Class'] == 1]
N = data[data['Class'] == 0]

P = P.drop(['Class'], axis=1)
N = N.drop(['Class'], axis=1)
data = data.drop(['Class'], axis=1)

data_array = data.values.tolist()
N_array = N.values.tolist()
P_array = P.values.tolist()

# {pnum}: 少数类数量  {nnum}: 多数类数量
pnum = len(P_array)
nnum = len(N_array)

# 对于整个P中求M个近邻
neigh = NearestNeighbors(n_neighbors=K + 1)
neigh.fit(P)

DANGER = []
SAFE = []
count = 0

# step 1
for i in range(pnum):
    M_ = 0
    nnarray = neigh.kneighbors([P_array[i]], K + 1, return_distance=False)
    nnarray_M = np.delete(nnarray[0], 0)

    for item in nnarray_M:
        if data_array[item] in N_array:
            M_ += 1

    # 生成
    if K > M_ >= (K / 2):
        DANGER.append(i)
    elif 0 <= M_ < K / 2:
        SAFE.append(i)
    else:
        count += 1

# print(f"DANGER = {DANGER}")
# print(f"len(DANGER) = {len(DANGER)}")
# print(f"SAFE = {SAFE}")
# print(f"len(SAFE) = {len(SAFE)}")
# print(count)
# print(pnum)

# print(P_array[DANGER[0]])
# print(P_array[SAFE[0]])

# 找出点
DANGER_array = []
SAFE_array = []

for item in range(len(DANGER)):
    # print(item)
    DANGER_array.append(P_array[DANGER[item]])

for item in range(len(SAFE)):
    # print(item)
    # P_array[item].append(1)
    SAFE_array.append(P_array[SAFE[item]])

# print(DANGER_array)
# print(SAFE_array)
# print(len(DANGER_array))
# print(len(SAFE_array))

# pinjie
# print(N_array)
# print(len(N_array))
# ada_array = N_array + DANGER_array
# rs_array = N_array + SAFE_array
# print(ada_array)
# print(len(ada_array))

DANGER_frame = pd.DataFrame(DANGER_array)
SAFE_frame = pd.DataFrame(SAFE_array)

DANGER_frame['Class'] = 1
SAFE_frame['Class'] = 1
# print(DANGER_frame)
# print(SAFE_frame)


N['Class'] = 0
# print(N)
N.columns = [0, 1, 2, 3, 4, 5, 6, 7, 'Class']

ada_frame = pd.concat([N, DANGER_frame], axis=0)
rs_frame = pd.concat([N, SAFE_frame], axis=0)
# print(ada_frame)
# print(rs_frame)

X_ada = ada_frame.iloc[:, ada_frame.columns != "Class"]
y_ada = ada_frame.iloc[:, ada_frame.columns == "Class"]

X_train, X_test, y_train, y_test = train_test_split(X_ada, y_ada, test_size=0.3, random_state=42)

X_adasyn, y_adasyn = adasyn(X_train, y_train)
maj_adasyn, min_adasyn = split(X_adasyn, y_adasyn)
# print(X_adasyn)
# print(y_adasyn)
syn_adasyn = duplicate_removal(min_adasyn, np.array(P))
# print(syn_adasyn)

X_train_rsmote = np.array(X_train)
y_train_rsmote = np.array(y_train).flatten()
X_rsmote, y_rsmote = rsmote(X_train_rsmote, y_train_rsmote)
maj_rsmote, min_rsmote = split(pd.DataFrame(X_rsmote), pd.DataFrame(y_rsmote, columns=['Class']))
# print(X_rsmote)
# print(y_rsmote)
# , columns=['Z1', 'Z2']
syn_rsmote = duplicate_removal(min_rsmote, np.array(P))
# print(syn_rsmote)

syn = syn_rsmote.tolist() + syn_adasyn.tolist()
# print(syn)
# print(len(syn))

syn_frame = pd.DataFrame(syn)
syn_frame['Class'] = 1
# print(syn_frame)

X_syn = syn_frame.iloc[:, syn_frame.columns != "Class"]
y_syn = syn_frame.iloc[:, syn_frame.columns == "Class"]
# print(len(X_syn))
# print(len(y_syn))

adars_dt_scores = DecisionTree("ADARS", X_syn, X_test, y_syn, y_test)
adars_rf_scores = RandomForest("ADARS", X_syn, X_test, y_syn, y_test)
# adars_svm_scores = SVM("ADARS", X_syn, X_test, y_syn, y_test_rs)
adars_nb_scores = NB("ADARS", X_syn, X_test, y_syn, y_test)
adars_knn_scores = KNN("ADARS", X_syn, X_test, y_syn, y_test)
# adars_lr_scores = LR("ADARS", X_syn, X_test, y_syn, y_test)
adars_ada_scores = Adaboost("ADARS", X_syn, X_test, y_syn, y_test)
