# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:36:16 2019

@author: zhuguohua
"""
########################################
## PC数据分析脚本，方向盘失控分析等
########################################
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

dat = np.loadtxt("../../PC_DataSet/2019_11_20/MIXed_3_1__BORUI_2019-11-20_15_11_39_MOVE.txt")

actual_angle_speed = dat[:,7]
actual = dat[:,8]
set_steering_angle = dat[:,90]

wheel_pulse = dat[:,76]

err_steering = []
for i in range(len(actual)):
    err_steering.append(set_steering_angle[i] - actual[i])

plt.close(1)
plt.figure(1)
plt.grid(1)
plt.plot(actual,linestyle="none", marker="*", linewidth=1.0)
plt.plot(set_steering_angle,linestyle="none", marker="*", linewidth=1.0)
#plt.plot(err_steering,linestyle="none", marker="*", linewidth=1.0)
plt.plot(actual_angle_speed,linestyle="none", marker="*", linewidth=1.0)
plt.show()

plt.close(2)
plt.figure(2)
plt.grid(2)
plt.plot(err_steering,linestyle="none", marker="*", linewidth=1.0)
plt.show()