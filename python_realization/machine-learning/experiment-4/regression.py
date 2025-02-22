'''
Description: Regression
Author: 张轩誉
Date: 2024-03-22 10:00:02
LastEditors: 张轩誉
LastEditTime: 2024-03-22 13:30:43
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 加载数据集
data = pd.read_csv('Folds5x2_pp.csv')

# 提取特征和标签
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# 最小二乘法线性回归
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# 使用SGDRegressor完成梯度下降
sgd_model = SGDRegressor(learning_rate='constant', eta0=0.001, max_iter=1000, random_state=42)
sgd_model.fit(X_train, y_train)

# 进行预测
ols_predictions = ols_model.predict(X_test)
sgd_predictions = sgd_model.predict(X_test)

# 绘制预测值与真实值的散点图
plt.figure(figsize=(10, 6))

plt.scatter(y_test, ols_predictions, label='OLS Predictions', alpha=0.3)
plt.scatter(y_test, sgd_predictions, label='SGD Predictions', alpha=0.2)
plt.scatter(y_test, y_test, label='True Label', alpha=1)

plt.xlabel('')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.legend()


# 评估模型
ols_mae = mean_absolute_error(y_test, ols_predictions)
ols_mse = mean_squared_error(y_test, ols_predictions)
sgd_mae = mean_absolute_error(y_test, sgd_predictions)
sgd_mse = mean_squared_error(y_test, sgd_predictions)

# 输出线性回归的系数
print("OLS Model Coefficients:", ols_model.coef_)
print("SGD Model Coefficients:", sgd_model.coef_)
# 输出线性回归的截距
print("OLS Model Intercept:", ols_model.intercept_)
print("SGD Model Intercept:", sgd_model.intercept_)

# 输出评价指标
print("OLS Model MAE:", ols_mae)
print("SGD Model MAE:", sgd_mae)
print("OLS Model MSE:", ols_mse)
print("SGD Model MSE:", sgd_mse)
print("OLS Model Score:", ols_model.score(X_test, y_test))
print("SGD Model Score:", sgd_model.score(X_test, y_test))

plt.show()
