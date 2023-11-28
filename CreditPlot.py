import numpy as np
import matplotlib.pyplot as plt
# 设置字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 正常显示负号
plt.rcParams["axes.unicode_minus"] = False

labels = ['F1', 'G-mean']
smo_result = [0.5930, 0.7097]
bs1_result = [0.5848, 0.7031]
bs2_result = [0.5917, 0.7091]
ada_result = [0.5723, 0.6921]
tl_result = [0.4625, 0.5684]
st_result = [0.5898, 0.7062]
ours_result = [0.5989, 0.7157]

x = np.arange(len(labels))
width = 0.1

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*3, smo_result, width, label='SMO', color='white', alpha=1, edgecolor="black", hatch='///')
rects2 = ax.bar(x - width*2, bs1_result, width, label='BS1', color='white', alpha=1, edgecolor="black", hatch='***')
rects3 = ax.bar(x - width, bs2_result, width, label='BS2', color='white', alpha=1, edgecolor="black", hatch='xxx')
rects4 = ax.bar(x, ada_result, width, label='ADA', color='black', alpha=0.25)
rects5 = ax.bar(x + width, tl_result, width, label='TL', color='black', alpha=0.5)
rects6 = ax.bar(x + width*2, st_result, width, label='ST', color='black', alpha=0.75)
rects7 = ax.bar(x + width*3, ours_result, width, label='本文', color='black')

ax.set_xticks(x, labels)
ax.legend(bbox_to_anchor=(0.5, -0.15), loc=8, ncol=10)

fig.tight_layout()
plt.savefig("CreditPlot.png", bbox_inches='tight', pad_inches=0.2, dpi=300)
plt.show()
