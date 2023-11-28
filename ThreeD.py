import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xx = np.arange(1, 10, 0.5)
yy = np.arange(1, 10, 0.5)
X, Y = np.meshgrid(xx, yy)
Z = X / Y

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.set_xlabel('$\it{nnum-pnum}$')
ax.set_ylabel(r'$\it{dnum}$')
ax.set_zlabel(r'$\delta$')
plt.savefig("ThreeD.jpg", bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()
