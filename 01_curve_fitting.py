# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:13:19 2016

@author: 王晓捷 11521053
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import least_square_fitting as lsf


def real_func(x):
    return np.sin(2*np.pi*x)

def show_chart(x_show):
    plt.plot(x_show,real_func(x_show),label='real sin')
    plt.plot(x_show,lsf.fitted_func(w,x_show),label='fitted curve')
    plt.plot(x,y_noise,'bo',label='with noise')
    plt.legend()
    plt.show()
    return 

x_show = np.linspace(0,1,1000)

"""
10 samples m = 4
"""  
w_num =4    
x = np.linspace(0,1,10)
y_real = real_func(x)
y_noise = [np.random.normal(0,0.3) + y for y in y_real]
w = lsf.least_square(x,y_noise,w_num)
print 'fit degree 4 in 10 samples, Fitting Parameters : ', w
show_chart(x_show)


"""
10 samples m = 10
"""  
w_num =10    
x = np.linspace(0,1,10)
y_real = real_func(x)
y_noise = [np.random.normal(0,0.3) + y for y in y_real]
w = lsf.least_square(x,y_noise,w_num)
print 'fit degree 10 in 10 samples, Fitting Parameters : ', w
show_chart(x_show)


"""
15 samples m = 10
"""  
w_num =10    
x = np.linspace(0,1,15)
y_real = real_func(x)
y_noise = [np.random.normal(0,0.3) + y for y in y_real]
w = lsf.least_square(x,y_noise,w_num)
print 'fit degree 10 in 15 samples, Fitting Parameters : ', w
show_chart(x_show)

"""
100 samples m = 10
"""  
w_num =10    
x = np.linspace(0,1,100)
y_real = real_func(x)
y_noise = [np.random.normal(0,0.3) + y for y in y_real]
w = lsf.least_square(x,y_noise,w_num)
print 'fit degree 10 in 100 samples, Fitting Parameters : ', w
show_chart(x_show)


"""
10 samples m = 10 with regularizatioin
"""  
w_num =10    
x = np.linspace(0,1,10)
y_real = real_func(x)
y_noise = [np.random.normal(0,0.3) + y for y in y_real]
lamda = math.exp(-18)
w = lsf.least_square_regularization(x,y_noise,w_num,lamda)
print 'fit degree 10 in 10 samples with regularization, Fitting Parameters : ', w
show_chart(x_show)