import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成一些模拟数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建RadiusNeighborsRegressor实例
radius = 0.5  # 设置半径
rnr = RadiusNeighborsRegressor(radius=radius, weights='distance')

# 训练模型
rnr.fit(X_train, y_train)

# 进行预测
y_pred = rnr.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 绘制真实值和预测值
plt.scatter(X_test, y_test, color='black', label='True values')
plt.scatter(X_test, y_pred, color='red', label='Predictions')

# 添加图例
plt.legend()

# 显示图表
plt.show()