# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:26:13 2019

@author: zhuguohua
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns 
import math as mt
###############################################################################
dat = np.loadtxt("../../NXP_DataSet/2019_5_23/20190523195347_进库_新库位_LocationMap_Data.txt")

#dat = np.loadtxt("../../NXP_DataSet/2019_5_7/20190507155403_入库1_LocationMap_Data.txt")
threshold_x_distance = 0.3
level_threadhold     = 2.2
###############################################################################
time_err = dat[:,0]
GroundLocation_x1 = dat[:,1]
GroundLocation_y1 = dat[:,2]
GroundLocation_s1 = dat[:,3]

GroundLocation_x2 = dat[:,4]
GroundLocation_y2 = dat[:,5]
GroundLocation_s2 = dat[:,6]

GroundLocation_x3 = dat[:,7]
GroundLocation_y3 = dat[:,8]
GroundLocation_s3 = dat[:,9]

GroundLocation_x4 = dat[:,10]
GroundLocation_y4 = dat[:,11]
GroundLocation_s4 = dat[:,12]

GroundLocation_x5 = dat[:,13]
GroundLocation_y5 = dat[:,14]
GroundLocation_s5 = dat[:,15]

GroundLocation_x6 = dat[:,16]
GroundLocation_y6 = dat[:,17]
GroundLocation_s6 = dat[:,18]

GroundLocation_x7 = dat[:,19]
GroundLocation_y7 = dat[:,20]
GroundLocation_s7 = dat[:,21]

GroundLocation_x8 = dat[:,22]
GroundLocation_y8 = dat[:,23]
GroundLocation_s8 = dat[:,24]

GroundLocation_x9 = dat[:,25]
GroundLocation_y9 = dat[:,26]
GroundLocation_s9 = dat[:,27]

GroundLocation_x10 = dat[:,28]
GroundLocation_y10 = dat[:,29]
GroundLocation_s10 = dat[:,30]

GroundLocation_x11 = dat[:,31]
GroundLocation_y11 = dat[:,32]
GroundLocation_s11 = dat[:,33]

GroundLocation_x12 = dat[:,34]
GroundLocation_y12 = dat[:,35]
GroundLocation_s12 = dat[:,36]

TrackPoint_x   = dat[:,37]
TrackPoint_y   = dat[:,38]
TrackPoint_yaw = dat[:,39]

LRU9_UltrasonicDistance1 = dat[:,40]
LRU9_UltrasonicDistance2 = dat[:,41]
LRU9_UltrasonicLevel     = dat[:,42]
LRU9_UltrasonicWidth     = dat[:,43]
LRU9_UltrasonicStatus    = dat[:,44]

LRU10_UltrasonicDistance1 = dat[:,45]
LRU10_UltrasonicDistance2 = dat[:,46]
LRU10_UltrasonicLevel     = dat[:,47]
LRU10_UltrasonicWidth     = dat[:,48]
LRU10_UltrasonicStatus    = dat[:,49]

LRU11_UltrasonicDistance1 = dat[:,50]
LRU11_UltrasonicDistance2 = dat[:,51]
LRU11_UltrasonicLevel     = dat[:,52]
LRU11_UltrasonicWidth     = dat[:,53]
LRU11_UltrasonicStatus    = dat[:,54]

LRU12_UltrasonicDistance1 = dat[:,55]
LRU12_UltrasonicDistance2 = dat[:,56]
LRU12_UltrasonicLevel     = dat[:,57]
LRU12_UltrasonicWidth     = dat[:,58]
LRU12_UltrasonicStatus    = dat[:,59]
###############################################################################
plt.close(1)
plt.figure(1)
plt.plot(GroundLocation_x11,GroundLocation_y11,linestyle="none", marker="*", linewidth=1.0)
plt.plot(GroundLocation_x12,GroundLocation_y12,linestyle="none", marker="*", linewidth=1.0)
plt.show()
###############################################################################
###############################################################################
#数值分布求取函数 该进版本v1
def  ValueDistributed_v1(step,dat):
    #获取数值范围
    value_max = max(dat)
    value_min = min(dat)
    print("最大值:",value_max)
    print("最小值:",value_min)
    array_cnt = int((value_max - value_min)/step) + 1
    print("分组数:",array_cnt)
    
    DistributedCnt = np.zeros(array_cnt)
    
    for v in range(len(dat)):
        for i in range(0,array_cnt):
            if (dat[v] >= (value_min + step * i)) and (dat[v] < (value_min + step * (i+1))):
                DistributedCnt[i] = DistributedCnt[i] + 1
    for i in range(0,array_cnt):           
        print("分布:",i,"，值：",DistributedCnt[i])
    list_distribute_max_cnt = DistributedCnt.tolist()
    distribute_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
    print("最高分布索引:",distribute_max_cnt)
    return value_min + step * distribute_max_cnt

#数值分布求取函数 该进版本v2
def  ValueDistributed_v2(step,dat):
    #获取数值范围
    value_max = max(dat)
    value_min = min(dat)
    print("最大值:",value_max)
    print("最小值:",value_min)
    array_cnt = int((value_max - value_min)/step) + 1
    print("分组数:",array_cnt)
    
    DistributedCnt = np.zeros(array_cnt)
    
    for v in range(len(dat)):
        for i in range(0,array_cnt):
            if (dat[v] >= (value_min + step * i)) and (dat[v] < (value_min + step * (i+1))):
                DistributedCnt[i] = DistributedCnt[i] + 1
    for i in range(0,array_cnt):           
        print("分布:",i,"，值：",DistributedCnt[i])
    list_distribute_max_cnt = DistributedCnt.tolist()
    distribute_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
    print("最高分布索引:",distribute_max_cnt)
    #######异常处理
#    if len(list_distribute_max_cnt) > 1:
#        list_distribute_max_cnt.remove(DistributedCnt[distribute_max_cnt])
##        del list_distribute_max_cnt[distribute_max_cnt]
#        distribute_sencond_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
#        print("第二高分布索引:",distribute_sencond_max_cnt)
    return value_min + step * distribute_max_cnt

# 缓存有效数据
def ValidDataProcess(datx,daty,level,level_threshold):   
    valid_process_x = []
    valid_process_y = []
    for i in range(len(level)):
        if level[i] > level_threshold:
            valid_process_x.append(datx[i])
            valid_process_y.append(daty[i])
    return valid_process_x,valid_process_y
  
def RedundancyDataProcess(v_d_x,v_d_y):
    redundancy_process_x = []
    redundancy_process_y = []
    last_value = v_d_x[0]
    for i in range(len(v_d_x)):
        if v_d_x[i] < last_value:
            redundancy_process_x.append(v_d_x[i])
            redundancy_process_y.append(v_d_y[i])
            if abs(v_d_x[0] - v_d_x[i]) > 2:
                break
            last_value = v_d_x[i]
    return redundancy_process_x,redundancy_process_y        
        
def FitDataProcess(datx,daty,threshold,step):
    valid_fit_x =[]
    valid_fit_y =[]
    for i in range(len(daty)):
        if (daty[i] >= threshold) and (daty[i] < (threshold + step)):
            valid_fit_x.append(datx[i])
            valid_fit_y.append(daty[i])
    return valid_fit_x,valid_fit_y

def LineFit(x,y):
    
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = 0
    for i in range(len(x)):
        sum_xy = sum_xy + x[i]*y[i]
    sum_x2 = 0
    for i in range(len(x)):
        sum_x2 = sum_x2 + x[i]*x[i]
    sum_n = len(x)
    
    Denominator = sum_n*sum_x2 - sum_x*sum_x
    
    molecule_b = sum_x2*sum_y - sum_x*sum_xy
    molecule_a = sum_n*sum_xy - sum_x*sum_y
    
    return molecule_a/Denominator,molecule_b/Denominator
    
###############################################################################
valid_process_11x,valid_process_11y = ValidDataProcess(GroundLocation_x11,GroundLocation_y11,LRU11_UltrasonicLevel,level_threadhold)
valid_process_12x,valid_process_12y = ValidDataProcess(GroundLocation_x12,GroundLocation_y12,LRU12_UltrasonicLevel,level_threadhold)  
###############################################################################
redundancy_data_11x,redundancy_data_11y = RedundancyDataProcess(valid_process_11x,valid_process_11y)
redundancy_data_12x,redundancy_data_12y = RedundancyDataProcess(valid_process_12x,valid_process_12y)
###############################################################################
plt.close(2)
plt.figure(2)
plt.plot(valid_process_11x,valid_process_11y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(valid_process_12x,valid_process_12y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(TrackPoint_x,TrackPoint_y,linestyle="none", marker="*", linewidth=1.0)
plt.show()

plt.close(3)
plt.figure(3)
plt.plot(redundancy_data_11x,redundancy_data_11y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(redundancy_data_12x,redundancy_data_12y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(TrackPoint_x,TrackPoint_y,linestyle="none", marker="*", linewidth=1.0)
plt.show()
###############################################################################
DistributeValue_11y = ValueDistributed_v2(0.05,redundancy_data_11y)
DistributeValue_12y = ValueDistributed_v2(0.05,redundancy_data_12y)

valid_fit_11x,valid_fit_11y = FitDataProcess(redundancy_data_11x,redundancy_data_11y,DistributeValue_11y,0.05)
valid_fit_12x,valid_fit_12y = FitDataProcess(redundancy_data_12x,redundancy_data_12y,DistributeValue_12y,0.05)
        
plt.close(4)
plt.figure(4)
plt.plot(valid_fit_11x,valid_fit_11y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(valid_fit_12x,valid_fit_12y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(TrackPoint_x,TrackPoint_y,linestyle="none", marker="*", linewidth=1.0)

###############################################################################
#直线方程函数
def f_1(x, A, B):
    return A*x + B

###############################################################################
A1, B1 = optimize.curve_fit(f_1, valid_fit_11x, valid_fit_11y)[0]
x1 = np.arange(-7,5,0.1)
y1 = A1 * x1 + B1
yaw11_line = mt.atan(A1)*57.3

A2, B2 = optimize.curve_fit(f_1, valid_fit_12x, valid_fit_12y)[0]
x2 = np.arange(-7,5,0.1)
y2 = A2 * x2 + B2
yaw12_line = mt.atan(A2)*57.3

x3 = np.arange(-7,5,0.1)
y3 = 0.5*(A1 + A2) * x3 + (B1 + B2)*0.5

plt.plot(x1,y1,linestyle="none", marker="*", linewidth=0.5,markersize=2)
plt.plot(x2,y2,linestyle="none", marker="*", linewidth=0.5,markersize=2)
plt.plot(x3,y3,linestyle="none", marker="*", linewidth=0.5,markersize=2)
plt.show()
###############################################################################
m_A1, m_B1 = LineFit(valid_fit_11x, valid_fit_11y)
