# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:50:37 2024

@author: 刘秋妙
"""

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

# 准备数据集    
df = pd.read_csv('house_price.csv', skiprows=1, nrows=100)
X = np.array(df.iloc[:,15])  #取一个特征量
y = np.array(df.iloc[:,-1])  #目标值

# 创建线性回归模型：y=mx+b
class linearRegression():
    # 初始化
    def __init__(self):
        super(linearRegression, self).__init__()
        
    def fit(self, X, y):  #运用最小二乘法求参数
        sum_Xy = 0
        sum_X = 0
        sum_y = 0
        sum_X2 = 0
        n = X.shape[0]
        for k in range(n):
            sum_Xy += X[k] * y[k]
            sum_X += X[k]
            sum_X2 += (X[k])**2
            sum_y += y[k]            
        self.m = (n*sum_Xy - sum_X * sum_y) / (n*sum_X2 - (sum_X)**2)
        self.b = (sum_y - self.m * sum_X) / n        
        return self.m, self.b
    
    def predict(self, X):  #线性回归曲线函数
        return self.m * X + self.b    

# 训练模型  
model = linearRegression()
model.fit(X, y)  

# 进行预测  
y_pred = model.predict(X)   
  
# 设置中文字体和负号显示  
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体  
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示问题  

# 可视化结果
plt.scatter(X, y, color='blue', label='真实值')    
plt.plot(X, y_pred, color='red', linewidth=2, label='预测值')  
plt.title('线性回归模型预测')  
plt.xlabel('自变量 X')  
plt.ylabel('因变量 y')  
plt.legend()  
plt.show()