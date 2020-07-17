# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:29:29 2019

@author: zhuguohua
"""
########################################################
## 超声波边界提取算法研究 边界提取和拟合算法
########################################################
import numpy as np
import pandas as pd
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

ultrasonic_beam_open_angle =  0.507098504392337

# beam open angle 58.11
def DataFusion(err_interval,err_d,beam_open_angle,ave_d):
    y = np.sqrt(err_interval*err_interval + err_d*err_d - 2*err_interval*err_d*np.cos(np.pi/2 - beam_open_angle/2))
    theta = err_d*np.sin(np.pi/2 - beam_open_angle/2)/y
    if theta > (beam_open_angle/2):
        D = ave_d*np.cos(theta - beam_open_angle/2)/np.cos(theta)
    else:
        theta = err_d*np.sin(np.pi/2)/err_interval
        D = ave_d/np.cos(theta)
    return D

def Vehicle2Ground(body_x,body_y,track_x,track_y,track_yaw):
    temp_x = body_x * np.cos(track_yaw) - body_y * np.sin(track_yaw)
    temp_y = body_x * np.sin(track_yaw) + body_y * np.cos(track_yaw)
    return track_x + temp_x,track_y + temp_y
    
def PositionRecalculate(pdat_x,pdat_y,distance):
    
    
    return
# line extracted algorithm
def LineExtracked(dat_x,dat_y):
    sum_x = 0
    sum_y = 0
    sum_xx = 0
    sum_yy = 0
    sun_xy = 0
    array_len = len(dat_x)
    
    for i in range(array_len):
        sum_x = sum_x + dat_x[i]
        sum_y = sum_y + dat_y[i]
        sum_xx = sum_xx + dat_x[i] * dat_x[i]
        sum_yy = sum_yy + dat_y[i] * dat_y[i]
        sun_xy = sun_xy + dat_x[i] * dat_y[i]
        
    V_x = sum_x / array_len
    V_y = sum_y / array_len
    V_xx = sum_xx - sum_x * sum_x / array_len
    V_yy = sum_yy - sum_y * sum_y / array_len
    V_xy = sun_xy - sum_x * sum_y / array_len
    
    err_Vxx_Vyy = V_xx - V_yy
    boot_value = np.sqrt(err_Vxx_Vyy*err_Vxx_Vyy + 4*V_xy*V_xy)
    theta = np.arctan((err_Vxx_Vyy - boot_value)/(2*V_xy))
    rhi = np.sin(theta)*V_x + np.cos(theta)*V_y
    E2 = (V_xx + V_yy - boot_value)/2
    S2 = E2/array_len
    return theta,rhi,E2,S2

# lines segment 
def LinesSegment(dat_x,dat_y):
    dat_set_x = []
    dat_set_y = []
    process_state = 0
    segment_theta = []
    segment_rhi   = []
    
    dat_set_x.clear()
    dat_set_y.clear()
    
    array_len = len(update_data_x)
    index_cnt = 0
    
    while(index_cnt < array_len):
        if process_state == 0:
            if len(dat_set_x) > nl:
                theta,rhi,E2,S2 = LineExtracked(dat_set_x,dat_set_y)
                if S2 < sl:
                    process_state = 1
                else:
                    del dat_set_x[0]
                    del dat_set_y[0]
                    dat_set_x.append(dat_x[index_cnt])
                    dat_set_y.append(dat_y[index_cnt])
                    index_cnt = index_cnt + 1
                    process_state = 0
            else:
                dat_set_x.append(dat_x[index_cnt])
                dat_set_y.append(dat_y[index_cnt])
                index_cnt = index_cnt + 1
                process_state = 0
       
        elif process_state == 1:
            e = np.abs(np.sin(theta)*dat_x[index_cnt] + np.cos(theta)*dat_y[index_cnt] - rhi)
            if e > n0*S2:
                segment_theta.append(theta)
                segment_rhi.append(rhi)    
                dat_set_x.clear()
                dat_set_y.clear()
                process_state = 0
            else:
                dat_set_x.append(dat_x[index_cnt])
                dat_set_y.append(dat_y[index_cnt])
                index_cnt = index_cnt + 1
                process_state = 2
                
        elif process_state == 2:
            theta,rhi,E2,S2 = LineExtracked(dat_set_x,dat_set_y)
            if S2 < sl:
                process_state = 1  
            else:
                process_state = 0 

    return segment_theta,segment_rhi
nl = 30
sl = 0.0005
n0 = 1

        
# Step1:计算更新点
update_data_x,update_data_y,update_data_level,update_data_width,update_data_distance1,update_data_distance2 = UpdateDataProcess(GroundLocation_x12,GroundLocation_y12,GroundLocation_s12,LRU12_UltrasonicLevel,LRU12_UltrasonicWidth,LRU12_UltrasonicDistance1,LRU12_UltrasonicDistance2)

# Step2: 直线提取
segment_theta,segment_rhi = LinesSegment(update_data_x,update_data_y)


lines_array_x = []
lines_array_y = []
for i in range(len(segment_theta)):
    lines_array_x.clear()
    lines_array_y.clear()
    for j in range(len(update_data_x)):
        lines_array_x.append(update_data_x[j])
        lines_array_y.append(-np.tan(segment_theta[i])*update_data_x[j] + segment_rhi[i]/np.cos(segment_theta[i]))
    
    
plt.close(1)
plt.figure(1)
plt.grid(1)
plt.plot(TrackPoint_x,TrackPoint_y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(TrackPoint_x,TrackPoint_yaw,linestyle="none", marker="*", linewidth=1.0)
plt.plot(update_data_x,update_data_y,linestyle="none", marker="*", linewidth=1.0)
plt.plot(update_data_x,update_data_distance1,linestyle="none", marker="*", linewidth=1.0)
for i in range(len(segment_theta)):
    lines_array_x.clear()
    lines_array_y.clear()
    for j in range(len(update_data_x)):
        lines_array_x.append(update_data_x[j])
        lines_array_y.append(-np.tan(segment_theta[i])*update_data_x[j] + segment_rhi[i]/np.cos(segment_theta[i]))
    plt.plot(lines_array_x,lines_array_y,linestyle="-", marker="*", linewidth=1.0)
plt.show()