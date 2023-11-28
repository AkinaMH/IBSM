import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
random.seed(42)


def BA_SMOTE(T, b=1, c=0.5, K=5):
    '''
    {T}: 总样本
    {P}: 少数类
    {N}: 多数类
    {K}: 从少数类minority最近邻，default=5
    '''
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

    # Lmaj = len(X[y == 1])
    # Lmin = len(X[y == 0])

    DANGER = []
    Ri = []

    # step 1 从少数类中求近邻
    neigh = NearestNeighbors(n_neighbors=K + 1)
    neigh.fit(T)

    for i in range(pnum):
        maj_num = 0
        # min_num = 0

        nnarray = neigh.kneighbors([P_array[i]], K + 1, return_distance=False)
        # print('nnarray', nnarray)

        # step 2
        for item in nnarray[0]:
            if T_array[item] in N_array:
                maj_num += 1

        if K / 2 <= maj_num and maj_num < K:
            DANGER.append(i)
            Ri.append(maj_num / K)

    # step 3 计算需要合成的样本总数G
    G = int(round((nnum - pnum) * b))

    # -------------------------------------------
    #     Z = sum(Ri)

    #     Synthetic = []
    #     for i in range(len(DANGER)):
    #         ri = Ri[i] / Z
    #         gi = round(ri * G)
    #         nnarray_minority = neigh.kneighbors([minority[DANGER[i]]], K, return_distance=False)
    #         for j in range(gi):
    #             k1 = random.choice(nnarray_minority[0])
    #             # print('nnarray_minority', nnarray_minority)
    #             # print('nnarray_minority[0]', nnarray_minority[0])
    #             list = nnarray_minority[0].tolist()
    #             list.remove(k1)
    #             k2 = random.choice(list)
    #             '''
    #             判断选择哪张插值方式
    #             1、k1和k2都是多
    #             2、k1少、k2多
    #             3、k1和k2都是少
    #             '''

    #             if (X[k1] in majority) and (X[k2] in majority):
    #                 # 插值1
    #                 xt = X[k1] + np.random.rand() * (X[k2] - X[k1])
    #                 synthetic = minority[DANGER[i]] + random.uniform(0, c) * (xt - minority[DANGER[i]])
    #                 Synthetic.append(synthetic)

    #             elif (X[k1] in minority) and (X[k2] in minority):
    #                 # 插值3
    #                 xt = X[k1] + np.random.rand() * (X[k2] - X[k1])
    #                 synthetic = minority[DANGER[i]] + np.random.rand() * (xt - minority[DANGER[i]])
    #                 Synthetic.append(synthetic)
    #             else:
    #                 # 插值2
    #                 xt = X[k1] + random.uniform(0, c) * (X[k2] - X[k1])
    #                 synthetic = minority[DANGER[i]] + np.random.rand() * (xt - minority[DANGER[i]])
    #                 Synthetic.append(synthetic)
    # -------------------------------------------

    # step 5 计算ri
    ri = []
    for item in Ri:
        ri.append(item / sum(Ri))

    # 计算每个少数类样本需要合成的数量gi
    gi = []
    for i in range(len(ri)):
        gi.append(round(ri[i] * G))
    # print(gi)

    # step 6 合成新样本
    Synthetic = []
    nnarray_minority = []

    for i in range(len(DANGER)):
        nnarray_minority = neigh.kneighbors([P_array[DANGER[i]]], K + 1, return_distance=False)

        for item in range(gi[i]):
            k1 = random.choice(nnarray_minority[0])
            # print('nnarray_minority', nnarray_minority)
            # print('nnarray_minority[0]', nnarray_minority[0])
            list = nnarray_minority[0].tolist()
            list.remove(k1)
            k2 = random.choice(list)
            '''
            判断选择哪张插值方式
            1、k1和k2都是多
            2、k1少、k2多
            3、k1和k2都是少
            '''
            if (T_array[k1] in N_array) and (T_array[k2] in N_array):
                # 插值1
                xt = T_array[k1] + np.random.rand() * (np.array(T_array[k2]) - np.array(T_array[k1]))
                synthetic = P_array[DANGER[i]] + random.uniform(0, c) * (xt - P_array[DANGER[i]])
                Synthetic.append(synthetic)

            elif (T_array[k1] in P_array) and (T_array[k2] in P_array):
                # 插值3
                xt = T_array[k1] + np.random.rand() * (np.array(T_array[k2]) - np.array(T_array[k1]))
                synthetic = P_array[DANGER[i]] + np.random.rand() * (xt - P_array[DANGER[i]])
                Synthetic.append(synthetic)
            else:
                # 插值2
                xt = T_array[k1] + random.uniform(0, c) * (np.array(T_array[k2]) - np.array(T_array[k1]))
                synthetic = P_array[DANGER[i]] + np.random.rand() * (xt - P_array[DANGER[i]])
                Synthetic.append(synthetic)

    Synthetic = np.array(Synthetic)

    return Synthetic
