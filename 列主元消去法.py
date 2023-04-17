# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:04:55 2023

@author: 86319
"""
#导入NumPy库并命名为np
import numpy as np

#定义高斯消元函数，输入为系数矩阵A和右侧常数向量b
def gauss_elimination(A, b):
    n = len(b) # 求出系数矩阵的行数，也就是未知数的个数
    # Gaussian elimination with partial pivoting（带有部分主元的高斯消元）
    for i in range(n-1):  # 针对每一列进行消元，最后一列不用处理
        # Find pivot row（找到主元所在的行）
        pivot_row = i  # 假设主元所在行为当前处理的行
        for j in range(i+1, n):  # 从当前行的下一行开始遍历
            if abs(A[j,i]) > abs(A[pivot_row,i]):  # 如果当前行的元素的绝对值大于主元所在行的对应元素绝对值
                pivot_row = j  # 将当前行作为主元所在的行
        # Swap rows（交换当前行和主元所在行）
        A[[i,pivot_row],i:] = A[[pivot_row,i],i:]  # 交换系数矩阵A的当前行和主元所在行
        b[[i,pivot_row]] = b[[pivot_row,i]]  # 交换常数向量b的当前元素和主元所在行的元素
        # Gaussian elimination（消元操作）
        for j in range(i+1, n):  # 针对当前列下面的每一行进行操作
            factor = A[j,i] / A[i,i]  # 计算当前行需要消元的位置的乘数
            A[j,i+1:] = A[j,i+1:] - factor * A[i,i+1:]  # 消元操作
            b[j] = b[j] - factor * b[i]  # 同时对常数向量进行操作

    # Back substitution（回代）
    x = np.zeros(n)  # 初始化解向量
    for i in range(n-1, -1, -1):  # 从最后一行开始往前
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]  # 回代求解出当前未知数
    
    return x

#使用给定的线性方程组进行测试
A = np.array([[-1, 2, -2], [3, -1, 4], [2, -3, -2]])
b = np.array([-1, 7, 0])
x = gauss_elimination(A, b)

#输出解向量
print("Solution: ", x)

