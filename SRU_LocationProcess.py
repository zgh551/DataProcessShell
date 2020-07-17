# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:15:06 2019

@author: zhuguohua
@descrip:用于三角定位的后库边确定
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns 


#dat = np.loadtxt("../../NXP_DataSet/2019_3_29/20190329175519_测试数据2_LocationMap_Data.txt")
#dat = np.loadtxt("../../NXP_DataSet/2019_4_12/20190412162334_LocationMap_Data.txt")
#dat = np.loadtxt("../../NXP_DataSet/2019_4_16/20190416164147_3.9_5.7_LocationMap_Data.txt")
#dat = np.loadtxt("../../NXP_DataSet/2019_4_16/20190416165118_x2.25y1.6_x4.85y1.4_LocationMap_Data.txt")
#dat = np.loadtxt("../../NXP_DataSet/2019_4_18/20190418151442_2_LocationMap_Data.txt")

dat = np.loadtxt("../../NXP_DataSet/2019_4_25/20190425105611_2_入库_LocationMap_Data.txt")

print(dat)
print(dat.shape)

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
###################################

###################################  
#直线方程函数
def f_1(x, A, B):
    return A*x + B
###################################
#front_edge_x = []
#front_edge_y = []
#rear_edge_x = []
#rear_edge_y = []
#last_x_value = 0
#for i in range(0,200): 
#    if GroundLocation_s11[i] == 0:
#        if GroundLocation_x11[i] < last_x_value:
#            front_edge_x.append(GroundLocation_x11[i])
#            front_edge_y.append(GroundLocation_y11[i])
#        last_x_value = GroundLocation_x11[i]
#for i in range(100,300): 
#    if GroundLocation_s6[i] == 0:
#        rear_edge_x.append(GroundLocation_x6[i])
#        rear_edge_y.append(GroundLocation_y6[i])
#    if GroundLocation_s7[i] == 0:
#        rear_edge_x.append(GroundLocation_x7[i])
#        rear_edge_y.append(GroundLocation_y7[i])
###################################   
#A1, B1 = optimize.curve_fit(f_1, front_edge_x, front_edge_y)[0]
#x1 = np.arange(-5,5,0.1)
#y1 = A1 * x1 + B1
#
#A2, B2 = optimize.curve_fit(f_1, rear_edge_x, rear_edge_y)[0]
#x2 = np.arange(-5,5,0.1)
#y2 = A2 * x2 + B2
###################################
#plt.close()
#
#plt.figure(1)
#plt.hist(front_edge_y, bins=100, color='steelblue', normed=True )
#sns.distplot(front_edge_y, rug=True)
###################################
#软件实现
#step_interval = 0.02
#distance_max = max(front_edge_y)
#distance_min = min(front_edge_y)
#print(distance_max)
#print(distance_min)
#array_cnt = int((distance_max - distance_min)/step_interval)
#print(array_cnt)
#
#DistributedCnt = np.zeros(array_cnt + 1)
#
#for v in range(len(front_edge_y)):
#    for i in range(0,array_cnt+1):
#        if (front_edge_y[v] >= (distance_min + step_interval * i)) and (front_edge_y[v] < (distance_min + step_interval * (i+1))):
#            DistributedCnt[i] = DistributedCnt[i] + 1
#for i in range(0,array_cnt):           
#    print(DistributedCnt[i])
#list_distribute_max_cnt = DistributedCnt.tolist()
#distribute_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
#print(distribute_max_cnt)
#
#left_boudary  = distribute_max_cnt;
#right_boudary = array_cnt + 1;
#while DistributedCnt[left_boudary] > DistributedCnt[distribute_max_cnt] * 0.65:
#    if left_boudary <= 0:
#        break;
#    else:
#        left_boudary = left_boudary - 1
#
#if left_boudary <= 0:
#    left_boudary = 0;
#else:
#    left_boudary = left_boudary + 1
#        
#while DistributedCnt[right_boudary] > DistributedCnt[distribute_max_cnt] * 0.65:
#    if right_boudary >= array_cnt :
#        break;
#    else:     
#        right_boudary = right_boudary + 1       
#        
#if right_boudary >= array_cnt: 
#    right_boudary = array_cnt
#else:
#    right_boudary = right_boudary - 1
#print(left_boudary)
#print(right_boudary)
###################################
#valid_front_edge_x = []
#valid_front_edge_y = []
#for i in range(len(front_edge_y)):
#    if (front_edge_y[i] >= (distance_min + step_interval * distribute_max_cnt)) and (front_edge_y[i] < (distance_min + step_interval * (distribute_max_cnt+1))):
#        valid_front_edge_x.append(front_edge_x[i])
#        valid_front_edge_y.append(front_edge_y[i])   
#print(len(front_edge_y))
#print(len(valid_front_edge_y))    
#A3, B3 = optimize.curve_fit(f_1, valid_front_edge_x, valid_front_edge_y)[0]
#x3 = np.arange(-5,5,0.1)
#y3 = A3 * x3 + B3
###################################
#plt.ion()
#plt.figure(2)
#plt.xlim(-12, 12)
#plt.ylim(-12, 12)
#plt.grid(color="k", linestyle=":")
#
##plt.plot(x1,y1)
##y1 = -1/A1 * x1 + B1
#plt.plot(x1,y1)
#plt.plot(x2,y2)
#plt.plot(x3,y3)
###################################
#数值分布求取函数
def  ValueDistributed(step,dat):
    #获取数值范围
    value_max = max(dat)
    value_min = min(dat)
    print("最大值:",value_max)
    print("最小值:",value_min)
    array_cnt = int((value_max - value_min)/step)
    print("分组数:",array_cnt)
    
    DistributedCnt = np.zeros(array_cnt + 1)
    
    for v in range(len(dat)):
        for i in range(0,array_cnt+1):
            if (dat[v] >= (value_min + step * i)) and (dat[v] < (value_min + step * (i+1))):
                DistributedCnt[i] = DistributedCnt[i] + 1
    for i in range(0,array_cnt):           
        print("分布:",i,"，值：",DistributedCnt[i])
    list_distribute_max_cnt = DistributedCnt.tolist()
    distribute_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
    print("最高分布索引:",distribute_max_cnt)
    print("分布值:",value_min + step * distribute_max_cnt)
    return value_min + step * distribute_max_cnt

#数值分布求取函数 该进版本v1
def  ValueDistributed_v1(step,dat):
    #获取数值范围
    value_max = max(dat)
    value_min = min(dat)
    print("最大值:",value_max)
    print("最小值:",value_min)
    array_cnt = int((value_max - value_min)/step)
    print("分组数:",array_cnt)
    
    DistributedCnt = np.zeros(array_cnt + 1)
    
    for v in range(len(dat)):
        for i in range(0,array_cnt+1):
            if (dat[v] >= (value_min + step * i)) and (dat[v] < (value_min + step * (i+1))):
                DistributedCnt[i] = DistributedCnt[i] + 1
    for i in range(0,array_cnt):           
        print("分布:",i,"，值：",DistributedCnt[i])
    list_distribute_max_cnt = DistributedCnt.tolist()
    distribute_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
    if distribute_max_cnt == 0:
        if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt + 1]) < 2:
            sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt + 1]
            master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
            slave_ratio  = DistributedCnt[distribute_max_cnt + 1] / sum_value
            distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt + 1.5)*slave_ratio)
            print("最高分布索引:",distribute_max_cnt,distribute_max_cnt+1) 
        else:
            distribute_value = value_min + step * (distribute_max_cnt + 0.5)
            print("最高分布索引:",distribute_max_cnt)
    elif distribute_max_cnt == (len(DistributedCnt) - 1):
        if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt - 1]) < 2:
            sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt - 1]
            master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
            slave_ratio  = DistributedCnt[distribute_max_cnt - 1] / sum_value
            distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt - 0.5)*slave_ratio)
            print("最高分布索引:",distribute_max_cnt,distribute_max_cnt - 1)  
        else:
            distribute_value = value_min + step * (distribute_max_cnt + 0.5)
            print("最高分布索引:",distribute_max_cnt)
    else:
        if DistributedCnt[distribute_max_cnt - 1] > DistributedCnt[distribute_max_cnt + 1]:
            if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt - 1]) < 2:
                sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt - 1]
                master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
                slave_ratio  = DistributedCnt[distribute_max_cnt - 1] / sum_value
                distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt - 0.5)*slave_ratio)
                print("最高分布索引:",distribute_max_cnt,distribute_max_cnt - 1) 
            else:
                distribute_value = value_min + step * (distribute_max_cnt + 0.5)
                print("最高分布索引:",distribute_max_cnt)
        else:
            if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt + 1]) < 2:
                sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt + 1]
                master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
                slave_ratio  = DistributedCnt[distribute_max_cnt + 1] / sum_value
                distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt + 1.5)*slave_ratio)
                print("最高分布索引:",distribute_max_cnt,distribute_max_cnt+1) 
            else:
                distribute_value = value_min + step * (distribute_max_cnt + 0.5)
                print("最高分布索引:",distribute_max_cnt)
    print("分布值:",distribute_value)            
    return distribute_value
#数值分布求取函数 返回众数数组
def  ValueDistributedArray(step,dat):
    valid_distribute_array = []
    #获取数值范围
    value_max = max(dat)
    value_min = min(dat)
    print("最大值:",value_max)
    print("最小值:",value_min)
    array_cnt = int((value_max - value_min)/step)
    print("分组数:",array_cnt)
    
    DistributedCnt = np.zeros(array_cnt + 1)
    
    for v in range(len(dat)):
        for i in range(0,array_cnt+1):
            if (dat[v] >= (value_min + step * i)) and (dat[v] < (value_min + step * (i+1))):
                DistributedCnt[i] = DistributedCnt[i] + 1
    for i in range(0,array_cnt):           
        print("分布:",i,"，值：",DistributedCnt[i])
    list_distribute_max_cnt = DistributedCnt.tolist()
    distribute_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
    print("最高分布索引:",distribute_max_cnt)
    print("分布值:",value_min + step * distribute_max_cnt)
    
    for v in range(len(dat)): 
        if (dat[v] >= (value_min + step * distribute_max_cnt)) and (dat[v] < (value_min + step * (distribute_max_cnt+1))):
            valid_distribute_array.append(dat[v]) 
    print(valid_distribute_array)
    return valid_distribute_array
###################################
    
#step_interval = 0.02
#distance_max = max(front_edge_y)
#distance_min = min(front_edge_y)
#print(distance_max)
#print(distance_min)
#array_cnt = int((distance_max - distance_min)/step_interval)
#print(array_cnt)
#
#DistributedCnt = np.zeros(array_cnt + 1)
#
#for v in range(len(front_edge_y)):
#    for i in range(0,array_cnt+1):
#        if (front_edge_y[v] >= (distance_min + step_interval * i)) and (front_edge_y[v] < (distance_min + step_interval * (i+1))):
#            DistributedCnt[i] = DistributedCnt[i] + 1
#for i in range(0,array_cnt):           
#    print(DistributedCnt[i])
#list_distribute_max_cnt = DistributedCnt.tolist()
#distribute_max_cnt = list_distribute_max_cnt.index(max(list_distribute_max_cnt))
#print(distribute_max_cnt)
#
#left_boudary  = distribute_max_cnt;
#right_boudary = array_cnt + 1;
#while DistributedCnt[left_boudary] > DistributedCnt[distribute_max_cnt] * 0.65:
#    if left_boudary <= 0:
#        break;
#    else:
#        left_boudary = left_boudary - 1
#
#if left_boudary <= 0:
#    left_boudary = 0;
#else:
#    left_boudary = left_boudary + 1
#        
#while DistributedCnt[right_boudary] > DistributedCnt[distribute_max_cnt] * 0.65:
#    if right_boudary >= array_cnt :
#        break;
#    else:     
#        right_boudary = right_boudary + 1       
#        
#if right_boudary >= array_cnt: 
#    right_boudary = array_cnt
#else:
#    right_boudary = right_boudary - 1
#print(left_boudary)
#print(right_boudary) 
###################################
plt.close(1)
plt.figure(1)
show_value = len(time_err)

min_x6 = 100
min_y6 = 100

min_x7 = 100
min_y7 = 100

valid_front_edge_x = []
valid_front_edge_y = []

valid_front_vehicle_edge_y = []

for i in range(show_value):
#    if GroundLocation_s5[i] == 0:
#        plt.plot(GroundLocation_x5[i],GroundLocation_y5[i],color="r", linestyle="none", marker="*", linewidth=1.0)
    if GroundLocation_s6[i] == 0:
        plt.plot(GroundLocation_x6[i],GroundLocation_y6[i],color="y", linestyle="none", marker="*", linewidth=1.0)
        valid_front_edge_x.append(GroundLocation_x6[i])
        valid_front_edge_y.append(GroundLocation_y6[i])
    if GroundLocation_s7[i] == 0:
        plt.plot(GroundLocation_x7[i],GroundLocation_y7[i],color="g", linestyle="none", marker="*", linewidth=1.0)
        valid_front_edge_x.append(GroundLocation_x7[i])
        valid_front_edge_y.append(GroundLocation_y7[i])
#    if GroundLocation_s8[i] == 0:
#        plt.plot(GroundLocation_x8[i],GroundLocation_y8[i],color="b", linestyle="none", marker="*", linewidth=1.0)
        
    if GroundLocation_s9[i] == 0:
        plt.plot(GroundLocation_x9[i],GroundLocation_y9[i],color="m", linestyle="none", marker="*", linewidth=1.0)
    if GroundLocation_s11[i] == 0:
        plt.plot(GroundLocation_x11[i],GroundLocation_y11[i],color="c", linestyle="none", marker="*", linewidth=1.0)
    
#    if GroundLocation_s10[i] == 0:
#        plt.plot(GroundLocation_x10[i],GroundLocation_y10[i],color="m", linestyle="none", marker="*", linewidth=1.0)
    if GroundLocation_s12[i] == 0:
        plt.plot(GroundLocation_x12[i],GroundLocation_y12[i],color="c", linestyle="none", marker="*", linewidth=1.0)
        valid_front_vehicle_edge_y.append(GroundLocation_y12)
    plt.plot(TrackPoint_x[i],TrackPoint_y[i],color="k", linestyle=":", marker="*", linewidth=1.0)
#    plt.pause(0.0001)
plt.show()

########################################################
# 数值分布求解
edge_point_x = ValueDistributed(0.1,valid_front_edge_x)
edge_point_y = ValueDistributed(0.1,valid_front_edge_y)
 
edge_point_x = ValueDistributed_v1(0.1,valid_front_edge_x)
edge_point_y = ValueDistributed_v1(0.1,valid_front_edge_y)

#valid_value_array_ditribute_x = []
#valid_value_array_ditribute_x = ValueDistributedArray(0.1,valid_front_vehicle_edge_y)
#y_data = np.arange(0,len(valid_value_array_ditribute_x),1)
#A1, B1 = optimize.curve_fit(f_1, valid_value_array_ditribute_x,x_data )[0]
#y1 = np.arange(-5,5,0.1)
##y1 = A1 * x1 + B1
#x1 = (y1 - B1)/A1
#
#valid_value_array_ditribute_y = []
#valid_value_array_ditribute_y = ValueDistributedArray(0.1,valid_front_edge_y)

plt.close(2)
plt.figure(2)

#plt.plot(x1,y1,color="r", linestyle="-", marker="*", linewidth=1.0,markersize=1)

plt.plot(edge_point_x,edge_point_y,color="r", linestyle="none", marker="*", linewidth=1.0,markersize=20)
plt.text(edge_point_x,edge_point_y+0.1,'%.2f' % edge_point_x , ha='center', va='bottom', fontsize=20)
for i in range(-10,10):
    plt.plot(i,edge_point_y,color="b", linestyle="--", marker="*", linewidth=1.0)
for i in range(-10,10):
    plt.plot(edge_point_x,i,color="b", linestyle="--", marker="*", linewidth=1.0)
    
show_value = len(time_err)

for i in range(show_value):
    if GroundLocation_s6[i] == 0:
        plt.plot(GroundLocation_x6[i],GroundLocation_y6[i],color="y", linestyle="none", marker="*", linewidth=1.0)
    if GroundLocation_s7[i] == 0:
        plt.plot(GroundLocation_x7[i],GroundLocation_y7[i],color="g", linestyle="none", marker="*", linewidth=1.0)
        
    if GroundLocation_s9[i] == 0:
        plt.plot(GroundLocation_x9[i],GroundLocation_y9[i],color="m", linestyle="none", marker="*", linewidth=1.0)
    if GroundLocation_s11[i] == 0:
        plt.plot(GroundLocation_x11[i],GroundLocation_y11[i],color="c", linestyle="none", marker="*", linewidth=1.0)
        
    if GroundLocation_s10[i] == 0:
        plt.plot(GroundLocation_x10[i],GroundLocation_y10[i],color="m", linestyle="none", marker="*", linewidth=1.0)
    if GroundLocation_s12[i] == 0:
        plt.plot(GroundLocation_x12[i],GroundLocation_y12[i],color="c", linestyle="none", marker="*", linewidth=1.0)
plt.show()