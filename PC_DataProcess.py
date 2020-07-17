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
dat = np.loadtxt("../../NXP_DataSet/2019_05_29/2019-05-29_19_48_59_MOVE_V.txt")
print(dat)
print(dat.shape)

time_err = dat[:,0]
#GroundLocation_x1 = dat[:,1]
#GroundLocation_y1 = dat[:,2]
#GroundLocation_s1 = dat[:,3]
#
#GroundLocation_x2 = dat[:,4]
#GroundLocation_y2 = dat[:,5]
#GroundLocation_s2 = dat[:,6]
#
#GroundLocation_x3 = dat[:,7]
#GroundLocation_y3 = dat[:,8]
#GroundLocation_s3 = dat[:,9]
#
#GroundLocation_x4 = dat[:,10]
#GroundLocation_y4 = dat[:,11]
#GroundLocation_s4 = dat[:,12]

GroundLocation_x5 = dat[:,21]
GroundLocation_y5 = dat[:,22]
GroundLocation_s5 = dat[:,23]

GroundLocation_x6 = dat[:,24]
GroundLocation_y6 = dat[:,25]
GroundLocation_s6 = dat[:,26]

GroundLocation_x7 = dat[:,27]
GroundLocation_y7 = dat[:,28]
GroundLocation_s7 = dat[:,29]

GroundLocation_x8 = dat[:,30]
GroundLocation_y8 = dat[:,31]
GroundLocation_s8 = dat[:,32]

GroundLocation_x9 = dat[:,33]
GroundLocation_y9 = dat[:,34]
GroundLocation_s9 = dat[:,35]

GroundLocation_x10 = dat[:,36]
GroundLocation_y10 = dat[:,37]
GroundLocation_s10 = dat[:,38]

GroundLocation_x11 = dat[:,39]
GroundLocation_y11 = dat[:,40]
GroundLocation_s11 = dat[:,41]

GroundLocation_x12 = dat[:,42]
GroundLocation_y12 = dat[:,43]
GroundLocation_s12 = dat[:,44]

TrackPoint_x   = dat[:,45]
TrackPoint_y   = dat[:,46]
TrackPoint_yaw = dat[:,47]

#LRU9_UltrasonicDistance1 = dat[:,40]
#LRU9_UltrasonicDistance2 = dat[:,41]
#LRU9_UltrasonicLevel     = dat[:,42]
#LRU9_UltrasonicWidth     = dat[:,43]
#LRU9_UltrasonicStatus    = dat[:,44]
#
#LRU10_UltrasonicDistance1 = dat[:,45]
#LRU10_UltrasonicDistance2 = dat[:,46]
#LRU10_UltrasonicLevel     = dat[:,47]
#LRU10_UltrasonicWidth     = dat[:,48]
#LRU10_UltrasonicStatus    = dat[:,49]
#
#LRU11_UltrasonicDistance1 = dat[:,50]
#LRU11_UltrasonicDistance2 = dat[:,51]
#LRU11_UltrasonicLevel     = dat[:,52]
#LRU11_UltrasonicWidth     = dat[:,53]
#LRU11_UltrasonicStatus    = dat[:,54]
#
#LRU12_UltrasonicDistance1 = dat[:,55]
#LRU12_UltrasonicDistance2 = dat[:,56]
#LRU12_UltrasonicLevel     = dat[:,57]
#LRU12_UltrasonicWidth     = dat[:,58]
#LRU12_UltrasonicStatus    = dat[:,59]
###################################
plt.close(1)
plt.figure(1)
#plt.plot(GroundLocation_x10,GroundLocation_y10,color="m", linestyle="none", marker="*", linewidth=1.0)
plt.plot(GroundLocation_x12,GroundLocation_y12,color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicLevel,color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,GroundLocation_s12,color="g", linestyle="none", marker="*", linewidth=1.0)
plt.show()
###############################################################################

    

#
#plt.close(2)
#plt.figure(2)
#plt.plot(GroundLocation_y12,color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LRU12_UltrasonicLevel,color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_s12,color="g", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
####################################
##边界值查找
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
##plt.plot(err_distance2,color="m", linestyle="--", marker="*", linewidth=1.0)
##plt.plot(err_level,color="g", linestyle="--", marker="*", linewidth=1.0)
#
#plt.plot(LRU12_UltrasonicDistance1,color="r", linestyle="none", marker="*", linewidth=1.0)
##plt.plot(LRU12_UltrasonicDistance2,color="c", linestyle="none", marker="*", linewidth=1.0)
##plt.plot(LRU12_UltrasonicLevel,color="b", linestyle="none", marker="*", linewidth=1.0)
#
#plt.plot(GroundLocation_s12,color="b", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
#
#plt.close(5)
#plt.figure(5)
#plt.plot(GroundLocation_x12,LRU12_UltrasonicDistance1,color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12,err_level,color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
#
#plt.close(6)
#plt.figure(6)
#index = 800
##plt.plot(GroundLocation_x10,GroundLocation_y10,color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],GroundLocation_y12[1:index],color="m", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],LRU12_UltrasonicDistance1[1:index],color="r", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],LRU12_UltrasonicDistance2[1:index],color="c", linestyle="none", marker="*", linewidth=1.0)
#plt.plot(GroundLocation_x12[1:index],LRU12_UltrasonicLevel[1:index],color="b", linestyle="none", marker="*", linewidth=1.0)
##plt.plot(GroundLocation_x12,GroundLocation_s12,color="g", linestyle="none", marker="*", linewidth=1.0)
#plt.show()
####################################
## 检测边界划分
#
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
####################################
## step :将距离为零的点消除，根据距离为0点的距离大小判定
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
##plt.plot(LRU12_UltrasonicWidth/100,color="y", linestyle="none", marker="*", linewidth=1.0)
##plt.plot(LRU12_UltrasonicWidth/LRU12_UltrasonicLevel,color="k", linestyle="none", marker="*", linewidth=1.0)
#plt.show() 
#
##第一步：找上升沿
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
