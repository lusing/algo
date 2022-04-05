from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# 创建 3D 图形对象
fig = plt.figure()
ax = Axes3D(fig)

X3 = np.linspace(-10,10,100)
Y3 = np.linspace(-10,10,100)
X3, Y3 = np.meshgrid(X3, Y3)
Z3 = X3*X3 + 0.2*Y3*Y3 + X3 + Y3 + 1
ax.plot_surface(X3, Y3, Z3, cmap=plt.cm.winter)

# 显示图
plt.show()
