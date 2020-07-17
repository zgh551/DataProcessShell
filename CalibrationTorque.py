# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:21:29 2019

@author: zhuguohua
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns 
###############################################################################
#dat = np.loadtxt("../../NXP_DataSet/2019_4_23/20190423153253_2_入库_LocationMap_Data.txt")
dat = np.loadtxt("../../NXP_DataSet/CalibrationTorque/20190723121606_80_Data.txt")

time_err = dat[:,0]

WheelSpeedRearLeftData = dat[:,1]
WheelSpeedRearRightData = dat[:,2]

LonAcc = dat[:,3]


wheel_middle_velocity = []
for i in range(len(time_err)):
    wheel_middle_velocity.append( (WheelSpeedRearLeftData[i] + WheelSpeedRearRightData[i]) * 0.5 )


plt.close(1)
plt.figure(1)
#plt.plot(WheelSpeedRearLeftData,LonAcc,linestyle="none", marker="*", linewidth=1.0)

plt.plot(WheelSpeedRearLeftData,linestyle="none", marker="*", linewidth=1.0)
plt.show()

plt.close(2)
plt.figure(2)
#plt.plot(WheelSpeedRearLeftData,LonAcc,linestyle="none", marker="*", linewidth=1.0)

plt.plot(LonAcc,linestyle="none", marker="*", linewidth=1.0)
plt.show()

acc_velocity = 0;
velocity_array = []
for i in range(len(time_err)):
    acc_velocity = acc_velocity + LonAcc[i] * 0.02
    velocity_array.append(acc_velocity)
    
plt.close(3)
plt.figure(3)
#plt.plot(WheelSpeedRearLeftData,LonAcc,linestyle="none", marker="*", linewidth=1.0)
plt.plot(wheel_middle_velocity,linestyle="none", marker="*", linewidth=1.0)
plt.plot(velocity_array,linestyle="none", marker="*", linewidth=1.0)
plt.show()