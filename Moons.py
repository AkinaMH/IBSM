import pandas as pd
from ImproveBorderline import *
from BA_SMOTE import *
from Sampling import *
from Plot import *
from sklearn.datasets import make_moons
import warnings

warnings.filterwarnings('ignore')


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


X, y = make_moons(n_samples=(650, 200), noise=0.2, random_state=42)

maj_ori = X[y == 0]
min_ori = X[y == 1]

plt.figure(figsize=(25, 10))
plt.subplot(2, 5, 1)
plot2class(maj_ori, min_ori, "(a) Original")
# plt.show()

# SMOTE
X_smote, y_smote = smote(X, y)

maj_smote, min_smote = split(pd.DataFrame(X_smote, columns=['Z1', 'Z2']), pd.DataFrame(y_smote, columns=['Class']))
# print(len(maj_smote))
# print(len(maj_ori))
# print(len(min_smote))
# print(len(min_ori))

# 留下生成样本
syn_smote = duplicate_removal(min_smote, min_ori)
# print(syn_smote)

plt.subplot(2, 5, 2)
plot3class(maj_ori, min_ori, syn_smote, "(b) SMOTE")
# plt.show()

# Borderline-SMOTE1
X_borderline, y_borderline = borderline(X, y)
maj_borderline, min_borderline = split(pd.DataFrame(X_borderline, columns=['Z1', 'Z2']), pd.DataFrame(y_borderline, columns=['Class']))

# 留下生成样本
syn_borderline = duplicate_removal(min_borderline, min_ori)
# print(syn_borderline)

plt.subplot(2, 5, 3)
plot3class(maj_ori, min_ori, syn_borderline, "(c) Borderline-SMOTE1")
# plt.show()

# Borderline-SMOTE2
X_borderline2, y_borderline2 = borderline2(X, y)
maj_borderline2, min_borderline2 = split(pd.DataFrame(X_borderline2, columns=['Z1', 'Z2']), pd.DataFrame(y_borderline2, columns=['Class']))

# 留下生成样本
syn_borderline2 = duplicate_removal(min_borderline2, min_ori)
# print(syn_borderline2)

plt.subplot(2, 5, 4)
plot3class(maj_ori, min_ori, syn_borderline2, "(d) Borderline-SMOTE2")
# plt.show()

# ADASYN
X_adasyn, y_adasyn = adasyn(X, y)
maj_adasyn, min_adasyn = split(pd.DataFrame(X_adasyn, columns=['Z1', 'Z2']), pd.DataFrame(y_adasyn, columns=['Class']))

# 留下生成样本
syn_adasyn = duplicate_removal(min_adasyn, min_ori)
# print(syn_adasyn)

plt.subplot(2, 5, 5)
plot3class(maj_ori, min_ori, syn_adasyn, "(e) ADASYN")
# plt.show()

# TomekLinks
# X_tomeklinks, y_tomeklinks = tomeklinks(X, y)
# maj_tomeklinks, min_tomeklinks = split(pd.DataFrame(X_tomeklinks, columns=['Z1', 'Z2']), pd.DataFrame(y_tomeklinks, columns=['Class']))
#
# # 留下生成样本
# syn_tomeklinks = duplicate_removal(maj_ori, maj_tomeklinks)
# # print(syn_tomeklinks)
# # 去除欠采样删除的点
# new_maj_ori_tomeklinks = duplicate_removal(maj_ori, syn_tomeklinks)
#
# plt.subplot(3, 3, 6)
# plot2class(new_maj_ori_tomeklinks, min_ori, "(f) TomekLinks")
# plt.show()

# SMOTETomek
X_smotetomek, y_smotetomek = smotetomek(X, y)
maj_smotetomek, min_smotetomek = split(pd.DataFrame(X_smotetomek, columns=['Z1', 'Z2']), pd.DataFrame(y_smotetomek, columns=['Class']))

# 留下过采样生成样本
syn_smotetomek_guo = duplicate_removal(min_smotetomek, min_ori)
# print(syn_smotetomek_guo)

# 去除欠采样删除的点
syn_smotetomek_qian = duplicate_removal(maj_ori, maj_smotetomek)
new_maj_ori_smotetomek = duplicate_removal(maj_ori, syn_smotetomek_qian)

plt.subplot(2, 5, 6)
plot3class(new_maj_ori_smotetomek, min_ori, syn_smotetomek_guo, "(f) SMOTETomek")
# plt.show()

# SMOTE-IPF
X_smoteipf, y_smoteipf = smote_ipf(np.array(X), np.array(y).flatten())
maj_smoteipf, min_smoteipf = split(pd.DataFrame(X_smoteipf, columns=['Z1', 'Z2']), pd.DataFrame(y_smoteipf, columns=['Class']))
# print(len(maj_smoteipf))
# print(len(maj_ori))
# print(len(min_smoteipf))
# print(len(min_ori))

# 留下生成样本
syn_smoteipf = duplicate_removal(min_smoteipf, min_ori)
# print(syn_smoteipf)

plt.subplot(2, 5, 7)
plot3class(maj_ori, min_ori, syn_smoteipf, "(g) SMOTE-IPF")
# plt.show()

# RSMOTE
X_rsmote, y_rsmote = rsmote(np.array(X), np.array(y).flatten())
maj_rsmote, min_rsmote = split(pd.DataFrame(X_rsmote, columns=['Z1', 'Z2']), pd.DataFrame(y_rsmote, columns=['Class']))
# print(len(maj_rsmote))
# print(len(maj_ori))
# print(len(min_rsmote))
# print(len(min_ori))

# 留下生成样本
syn_rsmote = duplicate_removal(min_rsmote, min_ori)
# print(syn_rsmote)

plt.subplot(2, 5, 8)
plot3class(maj_ori, min_ori, syn_rsmote, "(h) RSMOTE")
# plt.show()

# BA_SMOTE
train_ba = pd.concat([pd.DataFrame(X, columns=['Z1', 'Z2']), pd.DataFrame(y, columns=['Class'])], axis=1)
synthetic1 = BA_SMOTE(train_ba)

X_ba = np.r_[pd.DataFrame(X, columns=["Z1", "Z2"]), synthetic1]
y_ba = np.r_[np.array(pd.DataFrame(y, columns=["Class"]).values).flatten(), np.ones((synthetic1.shape[0]))]

X_ba = pd.DataFrame(X_ba, columns=["Z1", "Z2"])
y_ba = pd.DataFrame(y_ba, columns=["Class"])

maj_ba, min_ba = split(X_ba, y_ba)
# print(len(maj_ba))
# print(len(maj_ori))
# print(len(min_ba))
# print(len(min_ori))

# 留下生成样本
syn_ba = duplicate_removal(min_ba, min_ori)
# print(syn_ba)
# print(len(syn_ba))

plt.subplot(2, 5, 9)
plot3class(maj_ori, min_ori, syn_ba, "(i) BA-SMOTE")
# plt.show()

# ImproveBorderline
train = pd.concat([pd.DataFrame(X, columns=['Z1', 'Z2']), pd.DataFrame(y, columns=['Class'])], axis=1)
synthetic = ImproveBorderline(train)

X_improve = np.r_[pd.DataFrame(X, columns=['Z1', 'Z2']), synthetic]
y_improve = np.r_[np.array(pd.DataFrame(y, columns=['Class']).values).flatten(), np.ones((synthetic.shape[0]))]

X_column = ["Z1", "Z2"]
y_column = ["Class"]

X_improve = pd.DataFrame(X_improve, columns=X_column)
y_improve = pd.DataFrame(y_improve, columns=y_column)

maj_improve, min_improve = split(X_improve, y_improve)

# 留下生成样本
syn_improve = duplicate_removal(min_improve, min_ori)
# print(syn_improve)

plt.subplot(2, 5, 10)
plot3class(maj_ori, min_ori, syn_improve, "(j) ImproveBorderline")
# plt.show()
plt.legend(bbox_to_anchor=(-1.9, -0.3), loc=8, ncol=3, fontsize=15)

plt.savefig("Moons.JPG", bbox_inches='tight', pad_inches=0.2, dpi=600)
plt.show()

