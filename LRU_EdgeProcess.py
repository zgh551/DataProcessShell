# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:15:06 2019

@author: zhuguohua
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
# 4_18数据
dat = np.loadtxt("../../NXP_DataSet/2019_4_18/20190418152139_3_直线_LocationMap_Data.txt")
#dat = np.loadtxt("../../NXP_DataSet/2019_4_25/20190425105224_2_入库_LocationMap_Data.txt")


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
# 缓存有效数据
valid_process_x = []
valid_process_y = []
valid_process_id = []
valid_process_distance1 = []
valid_process_distance2 = []
valid_process_level = []

for i in range(len(time_err)):
#    if GroundLocation_s12[i] == 0:
    if LRU12_UltrasonicLevel[i] > 3:
        valid_process_x.append(GroundLocation_x12[i])
        valid_process_y.append(GroundLocation_y12[i])
        valid_process_id.append(i)
        valid_process_distance1.append(LRU12_UltrasonicDistance1[i])
        valid_process_distance2.append(LRU12_UltrasonicDistance2[i])
        valid_process_level.append(LRU12_UltrasonicLevel[i])

print("边界值：",min(valid_process_x))

min_value_x_axis = valid_process_x[0]
min_value_x_id = valid_process_id[0]
for i in range(len(valid_process_x)):
    if valid_process_x[i] < min_value_x_axis:
        min_value_x_axis = valid_process_x[i]
        min_value_x_id = valid_process_id[i]
        
print("计算出最小x轴坐标：",min_value_x_axis,"ID索引：",min_value_x_id)

#搜索level值为0的点
index_p = min_value_x_id
while LRU12_UltrasonicLevel[index_p] != 0:
    index_p = index_p + 1

print("边界点:",index_p)
print("边界坐标:",GroundLocation_x12[index_p - 1])
    
#通过距离值判定
last_valid_value_x_axis = valid_process_x[0]
err_data_valid = 0
#车辆信息
vehicle_first_edge  = 0
vehicle_second_edge = 0
vehicle_length      = 0
# 库位信息
parking_first_edge  = 0
parking_second_edge = 0
parking_length      = 0

edge_check_state = 0
threshold_x_distance = 0.3
boundary_value_id = 0
print("############################")
for i in range(len(valid_process_x)):
    err_data_valid = abs(valid_process_x[i] - last_valid_value_x_axis)    
    if edge_check_state == 0:
        if err_data_valid < threshold_x_distance:#点很密集，说明边沿连续
            vehicle_first_edge = last_valid_value_x_axis
            edge_check_state = 1
    elif edge_check_state == 1:
        if err_data_valid < threshold_x_distance:
            vehicle_second_edge = valid_process_x[i]  
            boundary_value_id   = valid_process_id[i]
        else: 
            ###############################################################
            vehicle_length = abs(vehicle_first_edge - vehicle_second_edge)
            print("车辆长度:",vehicle_length)
            print("车辆第一个边沿点：",vehicle_first_edge)
            print("车辆第二个边沿点：",vehicle_second_edge)
            print("############################")
            ###############################################################
            parking_first_edge  = last_valid_value_x_axis
            edge_check_state = 2 #进入下一点的边界判断
    elif edge_check_state == 2:
        if err_data_valid < threshold_x_distance and err_data_valid != 0:#密集点
            parking_second_edge = last_valid_value_x_axis
            vehicle_first_edge = last_valid_value_x_axis
            edge_check_state = 3
    elif edge_check_state == 3:
        if err_data_valid < threshold_x_distance:
            vehicle_second_edge = valid_process_x[i]
            boundary_value_id   = valid_process_id[i]
            if abs(vehicle_first_edge - vehicle_second_edge) > threshold_x_distance:#车辆边沿正确
                parking_length = abs(parking_first_edge - parking_second_edge)
                print("库位长度:",parking_length)
                print("库位第一个边沿点：",parking_first_edge)
                print("库位第二个边沿点：",parking_second_edge)
                print("############################")
                edge_check_state = 1
        else:#突然出现稀疏点
            parking_second_edge = valid_process_x[i]
            edge_check_state = 2
            
            
    last_valid_value_x_axis = valid_process_x[i]

if edge_check_state == 1:
    vehicle_length = abs(vehicle_first_edge - vehicle_second_edge)
    print("车辆长度:",vehicle_length)
    print("车辆第一个边沿点：",vehicle_first_edge)
    print("车辆第二个边沿点：",vehicle_second_edge)
    print("############################")
          
print("边界数据ID：",boundary_value_id)
#显示有效数据
plt.close(1)
plt.figure(1)
plt.plot(valid_process_x,valid_process_y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(valid_process_x,valid_process_distance1,linestyle="none", marker="*", linewidth=1.0)
plt.plot(valid_process_x,valid_process_distance2,linestyle="none", marker="*", linewidth=1.0)
plt.plot(valid_process_x,valid_process_level,linestyle="--", marker="*", linewidth=1.0)

plt.show()


#for i in range(len(valid_process_id)):
#    if valid_process_x()
plt.close(2)
plt.figure(2)
plt.plot(LRU12_UltrasonicDistance1,linestyle="none", marker="*", linewidth=1.0)
plt.plot(LRU12_UltrasonicDistance2,linestyle="none", marker="*", linewidth=1.0)
plt.plot(LRU12_UltrasonicDistance2 - LRU12_UltrasonicDistance1,linestyle="none", marker="*", linewidth=1.0)
plt.plot(LRU12_UltrasonicLevel,linestyle="none", marker="*", linewidth=1.0)
plt.show()


###################################
###################################
# step :将距离为零的点消除，根据距离为0点的距离大小判定
# 滤除距离值为0的异常点
def DataFilterLevel1(distance_dat,x_dat):
    over_fetch_start = 0
    over_fetch_end = 0
    next_fetch_start = 0
    next_fetch_end = 0
    state = 0
    last_distance = distance_dat[0]
    err_distance  = 0
    filter_distance_array = []
    for i in range(len(distance_dat)):
        err_distance = distance_dat[i] - last_distance
        filter_distance_array.append(distance_dat[i])
        if state == 0:#噪声点的起始点判定
            if (abs(err_distance) > 0.5):# and (distance_dat[i] == 0):
                next_fetch_end = i
                if abs(x_dat[over_fetch_start] - x_dat[over_fetch_end]) <= abs(x_dat[next_fetch_start] - x_dat[next_fetch_end]):
                    print("修正范围:",over_fetch_start,"-",over_fetch_end)
                    for id_index in range(over_fetch_start,over_fetch_end):
                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[over_fetch_start]
                else:
                    print("修正范围:",next_fetch_start,"-",next_fetch_end)
                    for id_index in range(next_fetch_start,next_fetch_end):
                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[next_fetch_start]
                over_fetch_start = i - 1; 
                state = 1
        elif state == 1:# 终点判定
            if (abs(err_distance) > 0.5):
                over_fetch_end = i;
                if abs(x_dat[over_fetch_start] - x_dat[over_fetch_end]) <= abs(x_dat[next_fetch_start] - x_dat[next_fetch_end]):
                    print("修正范围:",over_fetch_start,"-",over_fetch_end)
                    for id_index in range(over_fetch_start,over_fetch_end):
                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[over_fetch_start]
                else:
                    print("修正范围:",next_fetch_start,"-",next_fetch_end)
                    for id_index in range(next_fetch_start,next_fetch_end):
                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[next_fetch_start]
                next_fetch_start = i - 1
                state = 0
        elif state == 2:
            if (abs(err_distance) > 0.5):
                next_fetch_end = i
                if abs(x_dat[over_fetch_start] - x_dat[over_fetch_end]) <= abs(x_dat[next_fetch_start] - x_dat[next_fetch_end]):
                    print("修正范围:",over_fetch_start,"-",over_fetch_end)
                    for id_index in range(over_fetch_start,over_fetch_end):
                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[over_fetch_start]
                else:
                    print("修正范围:",next_fetch_start,"-",next_fetch_end)
                    for id_index in range(next_fetch_start,next_fetch_end):
                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[next_fetch_start]
                
#                if abs(x_dat[over_fetch_start] - x_dat[over_fetch_end]) > 0.3:#后期根据距离来判定
#                    print(over_fetch_start,"到",over_fetch_end,"是有效值")
#                else:
#                    print(over_fetch_start,"到",over_fetch_end,"是噪声")
#                    for id_index in range(over_fetch_start,over_fetch_end):
#                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[over_fetch_start]
                state = 0
#            if distance_dat[i] != 0:
#                over_fetch_end = i;
#                if abs(x_dat[over_fetch_start] - x_dat[over_fetch_end]) > 0.3:#后期根据距离来判定
#                    print(over_fetch_start,"到",over_fetch_end,"是有效值")
#                else:
#                    print(over_fetch_start,"到",over_fetch_end,"是噪声")
#                    for id_index in range(over_fetch_start,over_fetch_end):
#                        filter_distance_array[id_index] = LRU12_UltrasonicDistance1[over_fetch_start]
#                state = 0
                
                
#        elif state == 2:
#            if distance_dat[i] == 0:
#                if abs(x_dat[i] - x_dat[over_fetch_end]) < 0.1:
#                    for id_index in range(over_fetch_end,i):
#                        filter_distance_array[id_index] = 0
#                    state = 0
#            else:
#                if abs(x_dat[i] - x_dat[over_fetch_end]) > 0.1:
#                    if abs(x_dat[over_fetch_start] - x_dat[over_fetch_end]) > 0.2:#后期根据距离来判定
#                        print(over_fetch_start,"到",over_fetch_end,"是有效值")
#                    else:
#                        print(over_fetch_start,"到",over_fetch_end,"是噪声")
#                        for id_index in range(over_fetch_start,over_fetch_end):
#                            filter_distance_array[id_index] = LRU12_UltrasonicDistance1[over_fetch_start]
#                    state = 0
        last_distance = distance_dat[i]
    return filter_distance_array
###################################
def SinglePointValueFilter(distance_dat):
    over_fetch_start = 0
    over_fetch_end = 0
    state = 0
    last_distance = distance_dat[0]
    err_distance  = 0
    for i in range(len(distance_dat)):
        err_distance = distance_dat[i] - last_distance
        if state == 0:
            if (abs(err_distance) > 0.5) and (distance_dat[i] != 0):
                over_fetch_start = i - 1;   
                state = 1
        elif stae == 1:
            if (abs(err_distance) > 0.5):
                state = 0
    return
###################################
def DiscontinuityPointCheck(dat):
    last_distance = dat[0]
    err_distance  = 0
    start_point = 0
    end_point = 0
    
    edge_start_point_array = []
    edge_end_point_array = []
    for i in range(len(dat)):
        err_distance = dat[i] - last_distance
        if abs(err_distance) > 0.5:
            end_point = i-1
            edge_start_point_array.append(start_point)
            edge_end_point_array.append(end_point)
#            print("区间：",start_point,"<--->",end_point)
            start_point = i
        last_distance = dat[i]
    end_point = i
    edge_start_point_array.append(start_point)
    edge_end_point_array.append(end_point)
#    print("区间：",start_point,"<--->",end_point)
    return edge_start_point_array,edge_end_point_array
###################################
def DiscontinuityFilterMethod_V1(LRU12_UltrasonicDistance1,x_distance,start_array,end_array):
    filter_array = []
    for i in range(len(start_array)):
        if abs(x_distance[start_array[i]] - x_distance[end_array[i]]) > 0.5:
            for j in range(start_array[i],end_array[i]+1):
                filter_array.append(LRU12_UltrasonicDistance1[j])
        else:
            if i == 0:
                middle_cnt = end_array[i] - start_array[i]
                right_cnt  = end_array[i+1] - start_array[i+1]
                if middle_cnt >= right_cnt:
                    for j in range(start_array[i],end_array[i]+1):
                        filter_array.append(LRU12_UltrasonicDistance1[j])
                else:
                    for j in range(start_array[i],end_array[i]+1):
                        filter_array.append(LRU12_UltrasonicDistance1[start_array[i+1]])
            elif i == (len(start_array) - 1):           
                left_cnt   = end_array[i-1] - start_array[i-1]
                middle_cnt = end_array[i] - start_array[i]
                if middle_cnt > left_cnt:
                    for j in range(start_array[i],end_array[i]+1):
                        filter_array.append(LRU12_UltrasonicDistance1[j])
                else:
                    for j in range(start_array[i],end_array[i]+1):
                        filter_array.append(filter_array[end_array[i-1]])
            else:
                left_cnt   = end_array[i-1] - start_array[i-1]
                middle_cnt = end_array[i] - start_array[i]
                right_cnt  = end_array[i+1] - start_array[i+1]
                                   
                if (middle_cnt > left_cnt) or (middle_cnt > right_cnt):
                    for j in range(start_array[i],end_array[i]+1):
                        filter_array.append(LRU12_UltrasonicDistance1[j])
                else:
                    for j in range(start_array[i],end_array[i]+1):
                        filter_array.append(filter_array[end_array[i-1]])
                        
#                if (middle_cnt < left_cnt) and (middle_cnt < right_cnt):
#                    if left_cnt >= right_cnt:
#                        for j in range(start_array[i],end_array[i]+1):
#                            filter_array.append(filter_array[end_array[i-1]])
#                    else:
#                        for j in range(start_array[i],end_array[i]+1):
#                            filter_array.append(LRU12_UltrasonicDistance1[start_array[i+1]])
#                elif (middle_cnt >= left_cnt) and (middle_cnt < right_cnt): 
#                    for j in range(start_array[i],end_array[i]+1):
#                        filter_array.append(LRU12_UltrasonicDistance1[start_array[i+1]])
#                elif (middle_cnt < left_cnt) and (middle_cnt >= right_cnt):
#                     for j in range(start_array[i],end_array[i]+1):
#                         filter_array.append(filter_array[end_array[i-1]])
#                else:
#                     for j in range(start_array[i],end_array[i]+1):
#                        filter_array.append(LRU12_UltrasonicDistance1[j]) 
    return filter_array
###################################
    
LRU12_FilterData = []
#LRU12_FilterData = DataFilterLevel1(LRU12_UltrasonicDistance1,TrackPoint_x)

LRU12_EdgeStartArray = []
LRU12_EdgeEndArray = []
LRU12_EdgeStartArray,LRU12_EdgeEndArray = DiscontinuityPointCheck(LRU12_UltrasonicDistance1)

LRU12_FilterData = DiscontinuityFilterMethod_V1(LRU12_UltrasonicDistance1,TrackPoint_x,LRU12_EdgeStartArray,LRU12_EdgeEndArray)
# 捕捉边沿值
def EdgeFetch(dat):
    last_distance = dat[0]
    err_distance = np.zeros(len(dat))
    for i in range(len(dat)):
        err_distance[i] = dat[i] - last_distance
        if abs(err_distance[i]) < 0.5:
            err_distance[i] = 0
        else:                    
            if err_distance[i] > 0:
                if last_distance == 0:
                    print("下降沿ID:",i)
                else:
                    print("上升沿ID:",i) 
            else:
                if dat[i] == 0:
                    print("上升沿ID:",i)
                else:
                    print("下降沿ID:",i)                  
            err_distance[i] = abs(err_distance[i])
            
        last_distance = dat[i]
    return err_distance

err_edge_data = []
#err_edge_data = EdgeFetch(LRU12_FilterData)

plt.close(3)#滤波前后对比
plt.figure(3)
plt.plot(LRU12_UltrasonicDistance1,linestyle="none", marker="*", linewidth=1.0,markersize=15)
plt.plot(LRU12_FilterData,linestyle="none", marker="*", linewidth=1.0,markersize=10)
plt.plot(err_edge_data,linestyle="none", marker="*", linewidth=1.0,markersize=6)
plt.plot(LRU12_UltrasonicLevel,linestyle="none", marker="*", linewidth=1.0,markersize=6)
plt.show()










#plt.close(1)
#plt.figure(1)
#plt.plot(GroundLocation_x10,GroundLocation_y10,color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,GroundLocation_y12,color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicLevel,color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,GroundLocation_s12,color="g", linestyle="none", marker="*", linewidth=1.0)
#plt.show()

#plt.close(2)
#plt.figure(2)
#plt.plot(GroundLocation_y12,color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicLevel,color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_s12,color="g", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
###################################
#边界值查找
#last_distance = 0;
#err_distance1  = np.zeros(len(time_err));
#for i in range(len(time_err)):
#    err_distance1[i] = LRU12_UltrasonicDistance1[i] - last_distance
#    last_distance = LRU12_UltrasonicDistance1[i]
#    
#err_distance2  = np.zeros(len(time_err));   
#for i in range(len(time_err)):
#    err_distance2[i] = LRU12_UltrasonicDistance2[i] - last_distance
#    last_distance = LRU12_UltrasonicDistance2[i]
#
#err_level  = np.zeros(len(time_err));  
#for i in range(len(time_err)):
#    err_level[i] = LRU12_UltrasonicLevel[i] - last_distance
#    last_distance = LRU12_UltrasonicLevel[i]  
#     
#plt.close(3)
#plt.figure(3)
#plt.plot(err_distance1,color="m", linestyle="--", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicLevel,color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
#
#plt.close(4)
#plt.figure(4)
#plt.plot(err_distance1,color="y", linestyle="--", marker="*", linewidth=1.0)
#plt.plot(err_distance2,color="m", linestyle="--", marker="*", linewidth=1.0)
#plt.plot(err_level,color="g", linestyle="--", marker="*", linewidth=1.0)

#plt.plot(LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicLevel,color="b", linestyle="none", marker="*", linewidth=1.0)

#plt.plot(GroundLocation_s12,color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.show()

#plt.close(5)
#plt.figure(5)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicDistance1,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,err_level,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
#
#plt.close(6)
#plt.figure(6)
#index = 800
#plt.plot(GroundLocation_x10,GroundLocation_y10,color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],GroundLocation_y12[1:index],color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],LRU12_UltrasonicDistance1[1:index],color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],LRU12_UltrasonicDistance2[1:index],color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],LRU12_UltrasonicLevel[1:index],color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,GroundLocation_s12,color="g", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
###################################
# 检测边界划分

#for i in range(len(time_err)):
#    err_distance1[i] = LRU12_UltrasonicDistance1[i] - last_distance
#    
#    if err_distance1[i] < -0.5:
#        if LRU12_UltrasonicDistance1[i] == 0:
#            print("上升沿id:",i)
#        else:
#            print("下降沿id:",i)
#    elif err_distance1[i] > 0.5:
#        if last_distance != 0:
#            print("上升沿id:",i)
#        else:
#            print("下降沿id:",i)
#    last_distance = LRU12_UltrasonicDistance1[i]
###################################
# step :将距离为零的点消除，根据距离为0点的距离大小判定
#over_fetch_start = 0
#over_fetch_end = 0
#state = 0
#for i in range(len(time_err)):
#    if state == 0:#起始点判定
#        if LRU12_UltrasonicDistance1[i] == 0:
#            over_fetch_start = i;   
#            state = 1
#    elif state == 1:# 终点判定
#        if LRU12_UltrasonicDistance1[i] != 0:
#            over_fetch_end = i;
#            if (over_fetch_end - over_fetch_start) > 8:#后期根据距离来判定
#                print(over_fetch_start,"到",over_fetch_end,"是有效值")
#            else:
#                print(over_fetch_start,"到",over_fetch_end,"是噪声")
#                for id_index in range(over_fetch_start,over_fetch_end):
#                    LRU12_UltrasonicDistance1[id_index] = LRU12_UltrasonicDistance1[over_fetch_start-1]
#                    print(id_index)
#            state = 0
#
#last_distance = 0;
#err_distance1  = np.zeros(len(time_err));
#edge_state = 0
#up_edge_point = 0
#down_edge_point = 0
#
#for i in range(len(time_err)):
#    err_distance1[i] = LRU12_UltrasonicDistance1[i] - last_distance
#    
#    if err_distance1[i] < -0.5:
#        if LRU12_UltrasonicDistance1[i] == 0:
#            up_edge_point = i
#            print("上升沿id:",i)
#        else:
#            down_edge_point = i
#            print("下降沿id:",i)
#    elif err_distance1[i] > 0.5:
#        if last_distance != 0:
#            up_edge_point = i
#            print("上升沿id:",i)
#        else:
#            down_edge_point = i
#            print("下降沿id:",i)
#    
#    last_distance = LRU12_UltrasonicDistance1[i]
#    
#plt.close(7)
#plt.figure(7)
#plt.plot(LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(err_distance1,color="g", linestyle="--", marker="*", linewidth=1.0)
#plt.show()  
#
#revise_state = 0
#last_edge_point_cnt = 1
#for i in range(len(time_err)):
#    err_distance1[i] = LRU12_UltrasonicDistance1[i] - last_distance 
#    if revise_state == 0:
#        if abs(err_distance1[i]) > 0.5:
#            if (i - last_edge_point_cnt) < 8:
#                print("噪声范围：",last_edge_point_cnt,"到",i)
#                for id_index in range(last_edge_point_cnt,i):
#                    LRU12_UltrasonicDistance1[id_index] = LRU12_UltrasonicDistance1[last_edge_point_cnt-1]
#            last_edge_point_cnt = i
#    last_distance = LRU12_UltrasonicDistance1[i]
#
#last_distance = 0;
#err_distance1  = np.zeros(len(time_err));
#for i in range(len(time_err)):
#    err_distance1[i] = LRU12_UltrasonicDistance1[i] - last_distance
#    last_distance = LRU12_UltrasonicDistance1[i]
#    
#plt.close(8)#完全滤波后
#plt.figure(8)
#plt.plot(LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(err_distance1,color="g", linestyle="--", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicLevel,color="b", linestyle="--", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicWidth/100,color="y", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicWidth/LRU12_UltrasonicLevel,color="k", linestyle="none", marker="*", linewidth=1.0)
#plt.show() 

#第一步：找上升沿
#last_distance = LRU12_UltrasonicDistance1[0];
#err_distance1  = np.zeros(len(time_err));
#check_edge_state = 0
#up_edge_point = 0
#down_edge_point = 0
#last_level = 0
#
#search_tof2_up_point = 0
#search_level_min_up_point = 0
#
#search_tof2_down_point = 0
#search_level_min_down_point = 0
#
#for i in range(len(time_err)):
#    err_distance1[i] = LRU12_UltrasonicDistance1[i] - last_distance
#    
#    if check_edge_state == 0:  
#        if err_distance1[i] < -0.5:
#            if LRU12_UltrasonicDistance1[i] == 0:
#                up_edge_point = i
#                check_edge_state = 1
#                print("上升沿id:",i)
#        elif err_distance1[i] > 0.5:
#            if last_distance != 0:
#                up_edge_point = i
#                check_edge_state = 1
#                print("上升沿id:",i)
#    elif check_edge_state == 1:
#        search_tof2_up_point = up_edge_point
#        while LRU12_UltrasonicDistance2[search_tof2_up_point] == 0:
#            search_tof2_up_point = search_tof2_up_point - 1
#        print("上升沿二次回波边界点",search_tof2_up_point)
#        check_edge_state = 2
#    elif check_edge_state == 2:
#        search_level_min_up_point = search_tof2_up_point + 1
#        last_level = LRU12_UltrasonicLevel[search_level_min_up_point]
#        search_level_min_up_point = search_level_min_up_point + 1
#        while ((LRU12_UltrasonicLevel[search_level_min_up_point] - last_level) <= 0) and LRU12_UltrasonicLevel[search_level_min_up_point] != 0 and search_level_min_up_point < up_edge_point:
#            search_level_min_up_point = search_level_min_up_point + 1
#        search_level_min_up_point = search_level_min_up_point - 1
#        print("上升沿Level边界点",search_level_min_up_point)
#        print("上升沿实际推测的边界点",search_level_min_up_point)
#        print("上升沿实际推测的边界点坐标",GroundLocation_x12[search_level_min_up_point])
#        check_edge_state = 3#enter into down edge check
#    elif check_edge_state == 3:
#        if err_distance1[i] < -0.5:
#            if LRU12_UltrasonicDistance1[i] != 0:
#                down_edge_point = i
#                check_edge_state = 4
#                print("下降沿id:",i)
#        elif err_distance1[i] > 0.5:
#            if last_distance == 0:
#                down_edge_point = i
#                check_edge_state = 4
#                print("下降沿id:",i)
#    elif check_edge_state == 4:
#        search_tof2_down_point = down_edge_point
#        while (LRU12_UltrasonicDistance2[search_tof2_down_point] == 0) and (search_tof2_down_point < (len(LRU12_UltrasonicDistance2) - 1)):
#            search_tof2_down_point = search_tof2_down_point + 1
#        print("下降沿二次回波边界点",search_tof2_down_point)
#        check_edge_state = 5
#    elif check_edge_state == 5:   
#        search_level_min_down_point = search_tof2_down_point - 1
#        last_level = LRU12_UltrasonicLevel[search_level_min_down_point]
#        search_level_min_down_point = search_level_min_down_point - 1
#        while ((LRU12_UltrasonicLevel[search_level_min_down_point] - last_level) <= 0) and LRU12_UltrasonicLevel[search_level_min_down_point] != 0 and search_level_min_down_point < down_edge_point:
#            search_level_min_down_point = search_level_min_down_point - 1
#        search_level_min_down_point = search_level_min_down_point + 1
#        print("下降沿Level边界点",search_level_min_down_point)
#        print("下降沿实际推测的边界点",search_level_min_down_point)
#        print("下降沿实际推测的边界点坐标",GroundLocation_x12[search_level_min_down_point])
#        check_edge_state = 0
#    last_distance = LRU12_UltrasonicDistance1[i]
###################################  
