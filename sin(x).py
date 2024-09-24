# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:31:57 2024

@author: 刘秋妙
"""

import numpy as np  
import matplotlib.pyplot as plt  

class Polyfit5:  
    def __init__(self, x, y):  #初始化
        self.x = x  
        self.y = y  
    
    def polyfit(self):  #拟合参数
        n = self.x.shape[0]  
        # 创建 Vandermonde 矩阵  
        A = np.vstack([self.x**5, self.x**4, self.x**3, self.x**2, self.x, np.ones(n)]).T      
        # 使用最小二乘法求解系数  
        coefficients = np.linalg.inv(A.T @ A) @ A.T @ self.y
        # 解的顺序是 [a5, a4, a3, a2, a1, a0]  
        a5, a4, a3, a2, a1, a0 = coefficients     
        return a0, a1, a2, a3, a4, a5   
        
    def poly(self, x):  #拟合函数
        a0, a1, a2, a3, a4, a5 = self.polyfit()  
        return a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0  
    
# 生成数据点  
x = np.linspace(0, 2 * np.pi, 100)  # 从0到2π的100个点  
y = np.sin(x)

# 创建 Polyfit5 对象  
polyfit_model = Polyfit5(x, y)  

# 生成拟合曲线    
y_fit = polyfit_model.poly(x) 

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体  
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示问题

# 绘制图形  
plt.plot(x, y, label='sin(x)', color='blue')  # 原始正弦函数  
plt.plot(x, y_fit, label='sin(x)的五次模型拟合曲线', color='red', linestyle='--')  # 拟合曲线  
plt.title('sin(x)的五次模型拟合')  
plt.xlabel('x')  
plt.ylabel('y')  
plt.legend()  
plt.grid()  
plt.show()