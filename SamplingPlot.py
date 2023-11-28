import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from smote_variants import SMOTE

# 生成一个不平衡数据集
X, y = make_classification(
    n_classes=2, weights=[0.1, 0.9],
    n_informative=5, n_features=10, n_samples=100, random_state=42)

# 创建包含两个子图的图形
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 在第一个子图中展示原始数据分布
majority_class = X[y == 1]
minority_class = X[y == 0]
axes[0].scatter(majority_class[:, 0], majority_class[:, 1], color='blue', marker='o', s=25, edgecolor='k', label='Majority (Circle)')
axes[0].scatter(minority_class[:, 0], minority_class[:, 1], color='orange', marker='x', s=25, edgecolor='k', label='Minority (Cross)')
axes[0].set_title("Original Data Distribution")
# axes[0].legend()

# 使用 smote_variants 库中的 SMOTE 方法
oversampler = SMOTE()
X_resampled, y_resampled = oversampler.sample(X, y)

# 在第二个子图中展示过采样后的数据分布
majority_resampled = X_resampled[y_resampled == 1]
minority_resampled = X_resampled[y_resampled == 0]
axes[1].scatter(majority_resampled[:, 0], majority_resampled[:, 1], color='blue', marker='o', s=25, edgecolor='k', label='Circle')
axes[1].scatter(minority_resampled[:, 0], minority_resampled[:, 1], color='orange', marker='x', s=25, edgecolor='k', label='Cross')
axes[1].set_title("SMOTE Resampled Data Distribution")

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 打印过采样后的类别分布
unique, counts = np.unique(y_resampled, return_counts=True)
print("Class distribution after SMOTE:", dict(zip(unique, counts)))

# 保存图形到本地
fig.savefig('test.png')
