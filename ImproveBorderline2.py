import math
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors


np.random.seed(42)

def ImproveBorderline2(T, M=10, K=5):
    """
    {T}: 总样本
    {P}: 少数类
    {N}: 多数类
    {M}: 从T中求最近邻，default=10
    {K}: 从少数类P最近邻，default=5
    """
    # {P}: 少数类  {N}: 多数类
    P = T[T['Class'] == 1]
    N = T[T['Class'] == 0]

    P = P.drop(['Class'], axis=1)
    N = N.drop(['Class'], axis=1)
    T = T.drop(['Class'], axis=1)

    T_array = T.values.tolist()
    N_array = N.values.tolist()
    P_array = P.values.tolist()

    # {pnum}: 少数类数量  {nnum}: 多数类数量
    pnum = len(P_array)
    nnum = len(N_array)

    DANGER = []
    NOISE = []

    # 对于整个T中求M个近邻
    neigh = NearestNeighbors(n_neighbors=M + 1)
    neigh.fit(T)

    # step 1
    for i in range(pnum):
        M_ = 0
        nnarray = neigh.kneighbors([P_array[i]], M + 1, return_distance=False)
        nnarray_M = np.delete(nnarray[0], 0)

        for item in nnarray_M:
            if T_array[item] in N_array:
                M_ += 1

        # step 2
        if M > M_ >= (M / 2):
            # DANGER的下标存进去
            DANGER.append(i)
        # elif M_ == M:
        #     # NOISE的下标存进去
        #     NOISE.append(i)

    # 去除NOISE
    # for item in NOISE:
    #     P_array.remove(P_array[item])

    # for item in P_array:
        # if P_array[NOISE[i]] in P_array:
        # print('P_array', item)
        # else:

    # pnum = len(P_array)

    # step 3 对P求K个近邻
    # neigh_K = NearestNeighbors(n_neighbors=K + 1)
    # neigh_K.fit(P)

    dnum = len(DANGER)
    R = []
    c = (nnum - pnum) / dnum
    r = 1

    if c < 1:
        R = random.sample(DANGER, math.floor((c - 1) * dnum))
    else:
        R = DANGER
        r = math.floor(c + 0.5)

    if r > K:
        r = K

    R_len = len(R)

    Synthetic = []

    for i in range(R_len):
        nnarray = neigh.kneighbors([P_array[R[i]]], K + 1, return_distance=False)
        nnarray_K = np.delete(nnarray[0], 0)
        # print("nnarray_K", nnarray_K)

        # 计算前r个最短距离的坐标值
        dis = []
        for j in range(len(nnarray_K)):
            dis.append(abs(np.linalg.norm(np.array(P_array[R[i]]) - np.array(nnarray_K[j]))))

        # print('dis', dis)

        # 计算前r个最短距离的坐标值
        index = []
        nnarray_r = []
        for k in range(r):
            index.append(dis.index(min(dis)))
            nnarray_r.append(nnarray_K.tolist()[dis.index(min(dis))])
            dis.pop(dis.index(min(dis)))

        # nnarray_r = random.sample(nnarray_K.tolist(), r)
        # print(nnarray_r)

        for item in nnarray_r:
            # 属于少数类
            if T_array[item] in P_array:
                dif = np.array(T_array[item]) - np.array(P_array[R[i]])
                synthetic = P_array[R[i]] + dif * np.random.rand()
                Synthetic.append(synthetic)
            else:
                dif = np.array(T_array[item]) - np.array(P_array[R[i]])
                synthetic = P_array[R[i]] + dif * (np.random.rand()*0.5)
                Synthetic.append(synthetic)

    Synthetic = np.array(Synthetic)

    return Synthetic
