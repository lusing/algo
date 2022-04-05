import matplotlib.pyplot as plt
import numpy as np

import torch as t

# 生成x坐标的点
x = np.linspace(-10,10,1000)
# y是函数
y = t.sigmoid(t.tensor(x)).numpy()
plt.plot(x, y)

# 标题
plt.title("sigmoid")
# x轴描述
plt.xlabel("x")
# y轴描述
plt.ylabel("sigmoid(x)")
# 画网格线
plt.grid(True)
# 显示
plt.show()
