# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:21:58 2016
利用矩阵计算最小二乘，实现多项式拟合
主要公式是(。代表内积)
    拟合参数（无regularization） 
            w = inv(X.T。X)X.T。y
    拟合参数（有regularization） 
            w = inv(X.T。X + lamda。I)X.T。y
    拟合多项式函数值 y' = wx,调用poly1d即可
@author: 王晓捷 11521053
"""


import numpy as np
#get the value of fitted polynomial function: y = w0*x^(k-1) + w1*x^(k-2)+...+ wk
def fitted_func(w,x):
    f = np.poly1d(w)
    return f(x)
    
#get the matrix of X,like [1,x,x^2,x^3,....]    
def getX(x,fit_degree):
    w_num = fit_degree + 1
    x_mat = np.zeros((w_num,x.size))
    temp = np.ones_like(x)
    for i in range(0,w_num):    
        x_mat[w_num-i-1] = temp
        temp = x * temp
    x_mat  = x_mat.T
    return x_mat
    
def least_square(x,y,fit_degree):
    w_num = fit_degree + 1
    #get matix of X(x.size,w_num)
    x_mat = getX(x,w_num)   
    #计算x_mat.T与x_mat的内积，并求出内积方阵的逆矩阵
    square = np.dot(x_mat.T,x_mat)
    square_inv = np.linalg.inv(square)   
   
    w = np.dot(np.dot(square_inv,x_mat.T),y)
   
    return w.T   
    
def least_square_regularization(x,y,fit_degree,lamda):
    w_num = fit_degree
    #get matix of X(x.size,w_num)
    x_mat = getX(x,w_num)   
    #计算x_mat.T与x_mat的内积，并求出内积方阵的逆矩阵
    xx = np.dot(x_mat.T,x_mat)  
    i_mat = np.identity(xx.shape[0])
    square = xx + lamda * i_mat
    square_inv = np.linalg.inv(square)
    w = np.dot(np.dot(square_inv,x_mat.T),y)
   
    return w.T   
        