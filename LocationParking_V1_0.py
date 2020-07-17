# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:25:01 2019

@author: zhuguohua
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns 

#a = 4*4 + 95*95 - 92*92
#b = 2 * 4 * 95
#c = np.arccos(a/b)
###############################################################################
#dat = np.loadtxt("../../NXP_DataSet/2019_4_23/20190423153253_2_入库_LocationMap_Data.txt")
dat = np.loadtxt("../../NXP_DataSet/2019_4_25/20190425102457_1_入库_LocationMap_Data.txt")
#dat = np.loadtxt("../../NXP_DataSet/2019_5_7/20190507155403_入库1_LocationMap_Data.txt")
threshold_x_distance = 0.3
level_threadhold     = 2.2
credibility_threadhold = 0.5
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
# 根据原始数据更新最新的数据
def UpdateDataProcess(x,y,sts,level,width,distance1,distance2):
    update_process_x = []
    update_process_y = []
    update_process_level = []
    update_process_width = []
    update_process_distance1 = []
    update_process_distance2 = []
    last_x = 0
    for i in range(len(x)):
        if sts[i] == 0 :
            if last_x == 0:
                update_process_x.append(x[i])
                update_process_y.append(y[i])
                update_process_level.append(level[i])
                update_process_width.append(width[i]*0.01)
                update_process_distance1.append(distance1[i])
                update_process_distance2.append(distance2[i])
                last_x = x[i]
            else: 
                if(x[i] < last_x):
                    update_process_x.append(x[i])
                    update_process_y.append(y[i])
                    update_process_level.append(level[i])
                    update_process_width.append(width[i]*0.01)
                    update_process_distance1.append(distance1[i])
                    update_process_distance2.append(distance2[i])
                    last_x = x[i]
    return update_process_x,update_process_y,update_process_level,update_process_width,update_process_distance1,update_process_distance2

# 根据最新推送的数据，计算level的阀值
def AverageLevel(level):
    sum_dat = 0;
    length = len(level);
    for i in range(length):
        sum_dat = sum_dat + level[i];
    
    return sum_dat/length;
  
# 根据level阀值，处理更新点的数据
def ValidDataProcess(datx,daty,level,level_threshold):   
    valid_process_x = []
    valid_process_y = []
    for i in range(len(level)):
        if level[i] > level_threshold:
            valid_process_x.append(datx[i])
            valid_process_y.append(daty[i])
    return valid_process_x,valid_process_y 

def ValidDataProcessV1_1(datx,daty,level,level_threshold,width,width_threshold,d1,d2,d_threshold):   
    valid_process_x = []
    valid_process_y = []
    for i in range(len(level)):
        if d2[i] > d1[i]:
            if (d2[i] - d1[i]) < d_threshold:
                valid_process_x.append(datx[i])
                valid_process_y.append(daty[i])
        else:
            if level[i] > level_threshold:
                if width[i] > width_threshold:
                    valid_process_x.append(datx[i])
                    valid_process_y.append(daty[i])
    return valid_process_x,valid_process_y 

# 计算斜率
def DecliningCalculate(datx,daty):
    last_x = datx[0]
    last_y = daty[0]
    decline = []
    for i in range(len(datx)):
        decline.append((daty[i] - last_y)/(datx[i] - last_x))
        last_x = datx[i]
        last_y = daty[i]
    return decline


def WeightCalculation(dat):
    if dat < 0.5:
        return 4*(dat - 0.5)**2
    else:
        return 0
#    if dat < 0.5:
#        return (0.5- dat)/0.5
#    else:
#        return 0
    
def Distance(x,y,last_x,last_y):
    return ((x - last_x)**2 + (y - last_y)**2)**0.5


def DistanceCredibility(d1,d2):
    distance_credibility = []
    for i in range(len(d1)):
        distance_credibility.append(WeightCalculation(abs(d2[i] - d1[i])))
    return distance_credibility
# 数据点权值计算，判定数据有效性的概率,主要基于两点之间的距离
def DataCredibilityCalculate(datx,daty):
    credibility_value = []
    for i in range(len(datx)):
        if i == 0:
            credibility_value.append( (WeightCalculation(Distance(datx[i],daty[i],datx[i+1],daty[i+1])) + WeightCalculation(Distance(datx[i+1],daty[i+1],datx[i+2],daty[i+2])))*0.5 )
        elif i == 1:
            credibility_value.append( (WeightCalculation(Distance(datx[i],daty[i],datx[i-1],daty[i-1])) + WeightCalculation(Distance(datx[i],daty[i],datx[i+1],daty[i+1])) + WeightCalculation(Distance(datx[i+1],daty[i+1],datx[i+2],daty[i+2]))) / 3 )
        elif i == len(datx)-2:
            credibility_value.append( (WeightCalculation(Distance(datx[i],daty[i],datx[i+1],daty[i+1])) + WeightCalculation(Distance(datx[i],daty[i],datx[i-1],daty[i-1])) + WeightCalculation(Distance(datx[i-1],daty[i-1],datx[i-2],daty[i-2]))) / 3 )
        elif i == len(datx)-1:
            credibility_value.append( (WeightCalculation(Distance(datx[i],daty[i],datx[i-1],daty[i-1])) + WeightCalculation(Distance(datx[i-1],daty[i-1],datx[i-2],daty[i-2]))) * 0.5 )
        else:
            credibility_value.append( (WeightCalculation(Distance(datx[i+2],daty[i+2],datx[i+1],daty[i+1])) + WeightCalculation(Distance(datx[i],daty[i],datx[i+1],daty[i+1])) + WeightCalculation(Distance(datx[i],daty[i],datx[i-1],daty[i-1])) + WeightCalculation(Distance(datx[i-2],daty[i-2],datx[i-1],daty[i-1]))) * 0.25 )

    return credibility_value
###############################################################################
    
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
###############################################################################
def EdgeFinding_V1_0(valid_x,valid_y,threshold_distance):
    edge_check_state = 0
    err_data_valid = 0
    
    edge_state = 0
    #通过距离值判定
    last_valid_value_x_axis = valid_x[0]
    last_valid_value_y_axis = valid_y[0]
 
    vehicle_first_edge_x_temp = 0
    vehicle_first_edge_y_temp = 0
    vehicle_second_edge_x_temp = 0
    vehicle_second_edge_y_temp = 0
    
    #车辆信息
    vehicle_first_edge_x  = []
    vehicle_first_edge_y  = []
    vehicle_second_edge_x = []
    vehicle_second_edge_y = []
    
    # 库位信息
#    parking_first_edge_x  = []
#    parking_first_edge_y  = []
#    parking_second_edge_x = []
#    parking_second_edge_y = []
    print("############################")
    for i in range(len(valid_x)):
        err_data_valid = abs(valid_x[i] - last_valid_value_x_axis)    
        if 0 == edge_check_state:
            if err_data_valid > threshold_distance:
                vehicle_first_edge_x_temp = valid_x[i]
                vehicle_first_edge_y_temp = valid_y[i]
                edge_check_state = 1
            elif err_data_valid > 0:
                vehicle_first_edge_x_temp = last_valid_value_x_axis
                vehicle_first_edge_y_temp = last_valid_value_y_axis
                edge_check_state = 1
        elif 1 == edge_check_state:
            if err_data_valid > threshold_distance:
                edge_state = 0x55
                vehicle_second_edge_x_temp = last_valid_value_x_axis
                vehicle_second_edge_y_temp = last_valid_value_y_axis
                
                vehicle_first_edge_x.append(vehicle_first_edge_x_temp)
                vehicle_first_edge_y.append(vehicle_first_edge_y_temp)
                vehicle_second_edge_x.append(vehicle_second_edge_x_temp)
                vehicle_second_edge_y.append(vehicle_second_edge_y_temp)
                
                vehicle_first_edge_x_temp = valid_x[i]
                vehicle_first_edge_y_temp = valid_y[i]
            else:#如果一直密集，则更新车辆第二边沿位置坐标信息
                if abs(valid_x[i] - vehicle_first_edge_x_temp) > 0.5:
                    edge_state = 0xAA
                    vehicle_second_edge_x_temp = valid_x[i]
                    vehicle_second_edge_y_temp = valid_y[i]  
        last_valid_value_x_axis = valid_x[i]
        last_valid_value_y_axis = valid_y[i]
        
    if edge_state == 0xAA:
        vehicle_first_edge_x.append(vehicle_first_edge_x_temp)
        vehicle_first_edge_y.append(vehicle_first_edge_y_temp)
        vehicle_second_edge_x.append(vehicle_second_edge_x_temp)
        vehicle_second_edge_y.append(vehicle_second_edge_y_temp) 
    return vehicle_first_edge_x,vehicle_first_edge_y,vehicle_second_edge_x,vehicle_second_edge_y 


def EdgeFinding_V1_1(valid_x,valid_y,threshold_distance):
    edge_check_state = 0
    err_data_valid = 0
    
    edge_state = 0
    #通过距离值判定
    last_valid_value_x_axis = valid_x[0]
    last_valid_value_y_axis = valid_y[0]
 
    vehicle_first_edge_x_temp = 0
    vehicle_first_edge_y_temp = 0
    vehicle_second_edge_x_temp = 0
    vehicle_second_edge_y_temp = 0
    
    #车辆信息
    vehicle_first_edge_x  = []
    vehicle_first_edge_y  = []
    vehicle_second_edge_x = []
    vehicle_second_edge_y = []
    
    # 库位信息
#    parking_first_edge_x  = []
#    parking_first_edge_y  = []
#    parking_second_edge_x = []
#    parking_second_edge_y = []
    print("############################")
    for i in range(len(valid_x)):
        err_data_valid = abs(valid_x[i] - last_valid_value_x_axis)    
        if 0 == edge_check_state:
            if err_data_valid > threshold_distance:
                vehicle_first_edge_x_temp = valid_x[i]
                vehicle_first_edge_y_temp = valid_y[i]
                edge_check_state = 1
            elif err_data_valid > 0:
                vehicle_first_edge_x_temp = last_valid_value_x_axis
                vehicle_first_edge_y_temp = last_valid_value_y_axis
                edge_check_state = 1
        elif 1 == edge_check_state:
            if err_data_valid > threshold_distance:
                edge_state = 0x55
                vehicle_second_edge_x_temp = last_valid_value_x_axis
                vehicle_second_edge_y_temp = last_valid_value_y_axis
                
                vehicle_first_edge_x.append(vehicle_first_edge_x_temp)
                vehicle_first_edge_y.append(vehicle_first_edge_y_temp)
                vehicle_second_edge_x.append(vehicle_second_edge_x_temp)
                vehicle_second_edge_y.append(vehicle_second_edge_y_temp)
                
                vehicle_first_edge_x_temp = valid_x[i]
                vehicle_first_edge_y_temp = valid_y[i]
            else:#如果一直密集，则更新车辆第二边沿位置坐标信息
                if abs(valid_x[i] - vehicle_first_edge_x_temp) > 1:
                    edge_state = 0xAA
                    vehicle_second_edge_x_temp = valid_x[i]
                    vehicle_second_edge_y_temp = valid_y[i]  
        last_valid_value_x_axis = valid_x[i]
        last_valid_value_y_axis = valid_y[i]
        
    if edge_state == 0xAA:
        vehicle_first_edge_x.append(vehicle_first_edge_x_temp)
        vehicle_first_edge_y.append(vehicle_first_edge_y_temp)
        vehicle_second_edge_x.append(vehicle_second_edge_x_temp)
        vehicle_second_edge_y.append(vehicle_second_edge_y_temp) 
    return vehicle_first_edge_x,vehicle_first_edge_y,vehicle_second_edge_x,vehicle_second_edge_y 
###############################################################################
def CrossLine(x1,y1,x2,y2):
    theta = np.arctan2(x1 - x2,y2 -y1)
    r = x1 * np.cos(theta) + y1 * np.sin(theta)
    return r,theta
###############################################################################
# Step1:计算更新点
update_data_x,update_data_y,update_data_level,update_data_width,update_data_distance1,update_data_distance2 = UpdateDataProcess(GroundLocation_x12,GroundLocation_y12,GroundLocation_s12,LRU12_UltrasonicLevel,LRU12_UltrasonicWidth,LRU12_UltrasonicDistance1,LRU12_UltrasonicDistance2)
plt.close(1)
plt.figure(1)
plt.plot(update_data_x,update_data_y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(update_data_x,update_data_level,linestyle="none", marker="*", linewidth=1.0)
plt.plot(update_data_x,update_data_width,linestyle="none", marker="*", linewidth=1.0)
plt.plot(update_data_x,update_data_distance1,linestyle="none", marker="*", linewidth=1.0)
plt.plot(update_data_x,update_data_distance2,linestyle="none", marker="*", linewidth=1.0)
#area_level_width = []
#for i in range(len(update_data_level)):
#    area_level_width.append(update_data_level[i]*update_data_width[i]*0.5)
#plt.plot(update_data_x,area_level_width,linestyle="none", marker="*", linewidth=1.0)
#
#valid_average_area = AverageLevel(area_level_width);
#print("the average area value:",valid_average_area)
#
#valid_area_data_x,valid_area_data_y = ValidDataProcess(update_data_x,update_data_y,area_level_width,valid_average_area*0.8)
#plt.plot(valid_area_data_x,valid_area_data_y,linestyle="none", marker="*",color="b", linewidth=1.0,markersize=9)
###############################################################################
# Step2:计算平均Level值
valid_average_level = AverageLevel(update_data_level)
print("the average level value:",valid_average_level)

distribute_level = ValueDistributed(0.1,update_data_level)
print("the distribute level value:",distribute_level)
###############################################################################
# Step3:根据level值，计算有效的数据点
level_valid_data_x,level_valid_data_y = ValidDataProcessV1_1(update_data_x,update_data_y,update_data_level,valid_average_level,update_data_width,0.3,update_data_distance1,update_data_distance2,0.6)
plt.plot(level_valid_data_x,level_valid_data_y,linestyle="none", marker="*",color="r", linewidth=1.0)
###############################################################################
# Step4:根据二次回波，滤除边界点
#valid_distance_credibility = DistanceCredibility(update_data_distance1,update_data_distance2)
#
#plt.plot(update_data_x,valid_distance_credibility,linestyle="none", marker="*",color="y", linewidth=1.0)
###############################################################################
credibility_value_data = DataCredibilityCalculate(level_valid_data_x,level_valid_data_y)
#plt.plot(level_valid_data_x,credibility_value_data,linestyle="none", marker="*", linewidth=1.0)
###############################################################################
valid_average_credibility = AverageLevel(credibility_value_data);
print("the average credibility value:",valid_average_credibility)

valid_data_x,valid_data_y = ValidDataProcess(level_valid_data_x,level_valid_data_y,credibility_value_data,valid_average_credibility*0.66)
plt.plot(valid_data_x,valid_data_y,linestyle="none", marker="*",color="y", linewidth=1.0,markersize=3)
###############################################################################
#decline_value = DecliningCalculate(valid_data_x,valid_data_y)
#plt.plot(valid_data_x,decline_value,linestyle="--", marker="*", linewidth=1.0)
###############################################################################
valid_vehicle_first_x,valid_vehicle_first_y,valid_vehicle_second_x,valid_vehicle_second_y = EdgeFinding_V1_0(valid_data_x,valid_data_y,0.5)
for i in range(len(valid_vehicle_first_x)):
    plt.plot(valid_vehicle_first_x[i],valid_vehicle_first_y[i],linestyle="none", marker="*", linewidth=1.0,markersize=20)
    plt.text(valid_vehicle_first_x[i],valid_vehicle_first_y[i] + 0.1,'%.2f' % valid_vehicle_first_x[i] , ha='center', va='bottom', fontsize=20)

for i in range(len(valid_vehicle_second_x)):
    plt.plot(valid_vehicle_second_x[i],valid_vehicle_second_y[i],linestyle="none", marker="*", linewidth=1.0,markersize=20)
    plt.text(valid_vehicle_second_x[i],valid_vehicle_second_y[i] - 0.1,'%.2f' % valid_vehicle_second_x[i] , ha='center', va='bottom', fontsize=20)
    
plt.show()
# 缓存有效数据
#valid_process_x = []
#valid_process_y = []
#valid_process_id = []
#valid_process_distance1 = []
#valid_process_distance2 = []
#valid_process_level = []
#
#for i in range(len(time_err)):
#    if LRU12_UltrasonicLevel[i] > level_threadhold:
#        valid_process_x.append(GroundLocation_x12[i])
#        valid_process_y.append(GroundLocation_y12[i])
#        valid_process_id.append(i)
#        valid_process_distance1.append(LRU12_UltrasonicDistance1[i])
#        valid_process_distance2.append(LRU12_UltrasonicDistance2[i])
#        valid_process_level.append(LRU12_UltrasonicLevel[i])
############################################################################### 
#def EdgeFinding(valid_x,valid_y,threshold_distance):
#    edge_check_state = 0
#    #通过距离值判定
#    last_valid_value_x_axis = valid_x[0]
#    last_valid_value_y_axis = valid_y[0]
#    err_data_valid = 0
#    #车辆信息
#    vehicle_first_edge_x  = 0
#    vehicle_first_edge_y  = 0
#    vehicle_second_edge_x = 0
#    vehicle_second_edge_y = 0
#    # 库位信息
#    parking_first_edge_x  = 0
#    parking_first_edge_y  = 0
#    parking_second_edge_x = 0
#    parking_second_edge_y = 0
#    print("############################")
#    for i in range(len(valid_x)):
#        err_data_valid = abs(valid_x[i] - last_valid_value_x_axis)    
#        if edge_check_state == 0:
#            if err_data_valid < threshold_distance and err_data_valid != 0:#点很密集，说明边沿连续
#                vehicle_first_edge_x = last_valid_value_x_axis
#                vehicle_first_edge_y = last_valid_value_y_axis
#                print("#############车辆###############")
#                print("车辆第一个边沿点：",vehicle_first_edge_x,vehicle_first_edge_y)
#                edge_check_state = 1
#        elif edge_check_state == 1:
#            if err_data_valid < threshold_x_distance:
#                vehicle_second_edge_x = valid_x[i]  
#                vehicle_second_edge_y = valid_y[i] 
#            else: 
#                ###############################################################
#                print("#############车辆###############")
#                print("车辆第二个边沿点：",vehicle_second_edge_x,vehicle_second_edge_y)
#                ###############################################################
#                parking_first_edge_x  = last_valid_value_x_axis
#                parking_first_edge_y  = last_valid_value_y_axis
#                edge_check_state = 2 #进入下一点的边界判断
#        elif edge_check_state == 2:
#            if err_data_valid < threshold_x_distance and err_data_valid != 0:#密集点
#                parking_second_edge_x = last_valid_value_x_axis
#                parking_second_edge_y = last_valid_value_y_axis
#                vehicle_first_edge_x = last_valid_value_x_axis
#                vehicle_first_edge_y = last_valid_value_y_axis
#                edge_check_state = 3
#        elif edge_check_state == 3:
#            if err_data_valid < threshold_x_distance:
#                vehicle_second_edge_x = valid_x[i]  
#                vehicle_second_edge_y = valid_y[i]
#                if abs(vehicle_first_edge_x - vehicle_second_edge_x) > threshold_distance:#车辆边沿正确
#                    print("库位第一个边沿点：",parking_first_edge_x,parking_first_edge_y)
#                    print("库位第二个边沿点：",parking_second_edge_x,parking_second_edge_y)
#                    print("############################")
#                    edge_check_state = 1
#                else:
#                    
#            else:#突然出现稀疏点
#                parking_second_edge = valid_process_x[i]
#                edge_check_state = 2
#                
#                
#        last_valid_value_x_axis = valid_process_x[i]
#        last_valid_value_y_axis = valid_process_y[i]
#    if edge_check_state == 1:
#        boundary_value_id   = valid_process_id[i]
#        vehicle_length = abs(vehicle_first_edge - vehicle_second_edge)
#        print("车辆长度:",vehicle_length)
#        print("车辆第一个边沿点：",vehicle_first_edge)
#        print("车辆第二个边沿点：",vehicle_second_edge)
#        print("############################")
#              
#    print("边界数据ID：",boundary_value_id)
#    print("############################")
#    return boundary_value_id
          
###############################################################################
###############################################################################
###############################################################################


#数值分布求取函数 该进版本v1
def  ValueDistributed_v1(step,dat):
    #获取数值范围
    value_max = max(dat)
    value_min = min(dat)
    print("最大值:",value_max)
    print("最小值:",value_min)
    array_cnt = int((value_max - value_min)/step)
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
    if distribute_max_cnt == 0 and (len(DistributedCnt) - 1) > 1:
        if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt + 1]) < 2:
            sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt + 1]
            master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
            slave_ratio  = DistributedCnt[distribute_max_cnt + 1] / sum_value
            distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt + 1.5)*slave_ratio)
            print("最优分布索引:",distribute_max_cnt,distribute_max_cnt+1) 
        else:
            distribute_value = value_min + step * (distribute_max_cnt + 0.5)
            print("最优分布索引:",distribute_max_cnt)
    elif distribute_max_cnt == (len(DistributedCnt) - 1):
        if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt - 1]) < 2:
            sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt - 1]
            master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
            slave_ratio  = DistributedCnt[distribute_max_cnt - 1] / sum_value
            distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt - 0.5)*slave_ratio)
            print("最优分布索引:",distribute_max_cnt,distribute_max_cnt - 1)  
        else:
            distribute_value = value_min + step * (distribute_max_cnt + 0.5)
            print("最优分布索引:",distribute_max_cnt)
    else:
        if DistributedCnt[distribute_max_cnt - 1] > DistributedCnt[distribute_max_cnt + 1]:
            if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt - 1]) < 2:
                sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt - 1]
                master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
                slave_ratio  = DistributedCnt[distribute_max_cnt - 1] / sum_value
                distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt - 0.5)*slave_ratio)
                print("最优分布索引:",distribute_max_cnt,distribute_max_cnt - 1) 
            else:
                distribute_value = value_min + step * (distribute_max_cnt + 0.5)
                print("最优分布索引:",distribute_max_cnt)
        else:
            if (DistributedCnt[distribute_max_cnt] / DistributedCnt[distribute_max_cnt + 1]) < 2:
                sum_value = DistributedCnt[distribute_max_cnt] + DistributedCnt[distribute_max_cnt + 1]
                master_ratio = DistributedCnt[distribute_max_cnt] / sum_value
                slave_ratio  = DistributedCnt[distribute_max_cnt + 1] / sum_value
                distribute_value = value_min + step * ((distribute_max_cnt + 0.5)*master_ratio +  (distribute_max_cnt + 1.5)*slave_ratio)
                print("最优分布索引:",distribute_max_cnt,distribute_max_cnt+1) 
            else:
                distribute_value = value_min + step * (distribute_max_cnt + 0.5)
                print("最优分布索引:",distribute_max_cnt)
    print("分布值:",distribute_value) 
    print("#############################")           
    return distribute_value
###############################################################################
#plt.close(1)
#plt.figure(1)
#show_value = len(time_err)
#
#valid_front_edge_x = []
#valid_front_edge_y = []
#
#for i in range(show_value):
#    if GroundLocation_s6[i] == 0:
#        plt.plot(GroundLocation_x6[i],GroundLocation_y6[i],linestyle="none", marker="*", linewidth=1.0)
#        valid_front_edge_x.append(GroundLocation_x6[i])
#        valid_front_edge_y.append(GroundLocation_y6[i])
#    if GroundLocation_s7[i] == 0:
#        plt.plot(GroundLocation_x7[i],GroundLocation_y7[i],linestyle="none", marker="*", linewidth=1.0)
#        valid_front_edge_x.append(GroundLocation_x7[i])
#        valid_front_edge_y.append(GroundLocation_y7[i])
        
#    if GroundLocation_s5[i] == 0:
#        plt.plot(GroundLocation_x5[i],GroundLocation_y5[i],linestyle="none", marker="*", linewidth=1.0)
#        valid_front_edge_x.append(GroundLocation_x5[i])
#        valid_front_edge_y.append(GroundLocation_y5[i])
#        
#    if GroundLocation_s8[i] == 0:
#        plt.plot(GroundLocation_x8[i],GroundLocation_y8[i],linestyle="none", marker="*", linewidth=1.0)
#        valid_front_edge_x.append(GroundLocation_x8[i])
#        valid_front_edge_y.append(GroundLocation_y8[i])
#plt.show()
###############################################################################
# 数值分布求解
#edge_point_x = ValueDistributed_v1(0.1,valid_front_edge_x)
#edge_point_y = ValueDistributed_v1(0.1,valid_front_edge_y)
################################################################################
#plt.plot(edge_point_x,edge_point_y,color="r", linestyle="none", marker="*", linewidth=1.0,markersize=20)
#plt.text(edge_point_x,edge_point_y+0.1,'%.2f' % edge_point_x , ha='center', va='bottom', fontsize=20)
#for i in range(-10,10):
#    plt.plot(i,edge_point_y,color="b", linestyle="--", marker="*", linewidth=1.0)
#for i in range(-10,10):
#    plt.plot(edge_point_x,i,color="b", linestyle="--", marker="*", linewidth=1.0)
##显示有效数据
#plt.plot(valid_process_x,valid_process_y,linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[boundary_value_id],GroundLocation_y12[boundary_value_id],linestyle="none", marker="*", linewidth=1.0,markersize=20)
#plt.text(GroundLocation_x12[boundary_value_id],GroundLocation_y12[boundary_value_id]+0.1,'%.2f' % GroundLocation_x12[boundary_value_id] , ha='center', va='bottom', fontsize=20)
#plt.show()
###############################################################################
#plt.close(2)
#plt.figure(2)
#plt.plot(GroundLocation_x12,GroundLocation_y12,linestyle="none", marker="*", linewidth=1.0)
#plt.plot(valid_process_x,valid_process_y,linestyle="none", marker="*", linewidth=1.0)
#plt.show()
###############################################################################
    
###############################################################################
# 霍夫变换相关函数编写    
###############################################################################
# 计算三角函数查找表
def CreateTrigTable(numangle,min_theta,theta_step,irho):
    ang = min_theta
    tabSin = []
    tabCos = []
    for n in range(numangle):
        tabSin.append(np.sin(ang)*irho)
        tabCos.append(np.cos(ang)*irho)
        ang = ang + theta_step
    return tabSin,tabCos

# 查找局部极大值
def findLocalMaximums(numrho,numangle,threshold,accum):
    sort_buf = []
    for r in range(numrho):
        for n in range(numangle):
            base = (n + 1)*(numrho + 2) + r + 1
            if (accum[base] > threshold) and (accum[base] > accum[base - 1]) and (accum[base] >= accum[base + 1]) and (accum[base] > accum[base - numrho - 2]) and (accum[base] >= accum[base + numrho + 2]):
                   sort_buf.append(base) 
    return sort_buf

# 经典霍夫变换直线提取
def HoughLinesStandard(x,y,
                       rho,theta,
                       threshold,linesMax,
                       min_theta,max_theta):
    irho = 1 / rho
    dat_len = len(x)
    max_rho = max(np.sqrt(x[0]*x[0] + y[0]*y[0]),np.sqrt(x[dat_len - 1]*x[dat_len - 1] + y[dat_len - 1]*y[dat_len - 1]))
    min_rho = -max_rho
    
    numangle = int((max_theta - min_theta) / theta)
    numrho   = int((max_rho - min_rho + 1) / rho  )
    
    accum  = np.zeros(((numangle + 2) * (numrho + 2)),dtype = int)
    
    
    tabSin,tabCos = CreateTrigTable(numangle,min_theta,theta,irho)
    
    for i in range(dat_len):
        for n in range(numangle):
            r = int(x[i] * tabCos[n] + y[i] * tabSin[n]);
            r = int(r + (numrho - 1)/2)
            accum[(n + 1)*(numangle + 2) + r + 1] += 1
    #查找局部极大值
    line_base_buf = findLocalMaximums(numrho,numangle,threshold,accum)
    
    #排序
    # 目前求取最大计数的数据
    max_index_acc = 0
    max_acc_num   = 0
    for i in range(len(line_base_buf)):
        if accum[line_base_buf[i]] > max_acc_num:
            max_acc_num = accum[line_base_buf[i]]
            max_index_acc = line_base_buf[i]
#    max_index_acc = 13531
#    max_index_acc = 12086
#    max_index_acc = 12746
    # 将最大值的点存起来
    valid_x_array = []
    valid_y_array = []
    for i in range(dat_len):
        for n in range(numangle):
            r = int(x[i] * tabCos[n] + y[i] * tabSin[n]);
            r = int(r + (numrho - 1)/2)
            index = (n + 1)*(numangle + 2) + r + 1
            if index == max_index_acc:
                valid_x_array.append(x[i])
                valid_y_array.append(y[i])
    # 转换为直线
    scale = 1.0/(numrho + 2)
    
    n = int(max_index_acc * scale) - 1
    r = max_index_acc - (n + 1)*(numrho + 2) - 1
    
    line_rho = (r - (numrho - 1) *0.5 ) * rho
    line_angle = min_theta + n * theta
    
    return line_base_buf,accum,line_rho,line_angle,valid_x_array,valid_y_array
###############################################################################
# 对处理后的数据 update_data_x 和 update_data_y 进行霍夫变换 使用最新代码
i_data_y = []
for i in range(len(update_data_y)):
    i_data_y.append(-update_data_y[i])
    
base_buf,acc_buf,line_cal_rho,line_cal_angle,filter_x_array,filter_y_array = HoughLinesStandard(update_data_x,i_data_y,
                   0.05,0.03,
                   28,2,
                   0,np.pi)

###############################################################################
###############################################################################
#直线方程函数
def f_1(x, A, B):
    return A*x + B

###############################################################################
line_fit_i_temp = []
for i in range(len(filter_y_array)):
    line_fit_i_temp.append(-filter_y_array[i])
A1, B1 = optimize.curve_fit(f_1, filter_x_array, line_fit_i_temp)[0]
line_fit_x1 = np.arange(int(min(update_data_x))-1,int(max(update_data_x))+1,0.1)
line_fit_y1 = A1 * line_fit_x1 + B1

angle_line = np.arctan(0.09873896464964399)*57.3


A2, B2 = optimize.curve_fit(f_1, update_data_x, update_data_y)[0]
line_fit_x2 = np.arange(int(min(update_data_x))-1,int(max(update_data_x))+1,0.1)
line_fit_y2 = A2 * line_fit_x2 + B2
# 测试霍夫变换效果，思路不对
#r_arrary = []
#theta_arrary = []
#
#for i in range(len(update_data_x)-1):
#    for j in range(len(update_data_x)):
#        if i != j:
##            r_temp,theta_temp = CrossLine(update_data_x[i],update_data_y[i],update_data_x[i+1],update_data_y[i+1])
#            r_temp,theta_temp = CrossLine(update_data_x[i],update_data_y[i],update_data_x[j],update_data_y[j])
#            r_arrary.append(r_temp)
#            theta_arrary.append(theta_temp)
#
#r_distribute_value = ValueDistributed_v1(0.2,r_arrary)
#theta_distribute_value = ValueDistributed_v1(0.2,theta_arrary)
line_x = []
line_y = []
for i in range(int(min(update_data_x))-1,int(max(update_data_x))+1):
    line_x.append(i)
    line_y.append(-i/np.tan(line_cal_angle) + line_cal_rho/np.sin(line_cal_angle))
    
plt.close(2)
plt.figure(2)
plt.plot(acc_buf,linestyle="none", marker="*", linewidth=1.0)
plt.show()

plt.close(3)
plt.figure(3)
#plt.plot(acc_buf,linestyle="none", marker="*", linewidth=1.0)
plt.plot(base_buf,linestyle="none", marker="*", linewidth=1.0)
plt.show()

plt.close(4)
plt.figure(4)
plt.plot(update_data_x,update_data_y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(filter_x_array,filter_y_array,linestyle="none", marker="*", linewidth=1.0)
plt.plot(line_x,line_y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(line_fit_x1,line_fit_y1,linestyle="none", marker="*", linewidth=1.0)
plt.plot(line_fit_x2,line_fit_y2,linestyle="none", marker="*", linewidth=1.0)
plt.show()