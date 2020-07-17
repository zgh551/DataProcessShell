# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:30:19 2019

@author: zhuguohua
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns 
###############################################################################
dat = np.loadtxt("../../NXP_DataSet/CalibrationTorque_7_25/20190725154958_150_Data.txt")

time_err = dat[:,0]

WheelSpeedRearLeftData  = dat[:,1]
WheelSpeedRearRightData = dat[:,2]

WheelSpeedRearLeftPulse  = dat[:,3]
WheelSpeedRearRightPulse = dat[:,4]

VehicleSpeed             = dat[:,5]

LonAcc = dat[:,6]
LatAcc = dat[:,7]
YawRate = dat[:,8]

wheel_middle_velocity = []
for i in range(len(time_err)):
    wheel_middle_velocity.append( (WheelSpeedRearLeftData[i] + WheelSpeedRearRightData[i]) * 0.5 )


plt.close(1)
plt.figure(1)

plt.plot(wheel_middle_velocity,linestyle="none", marker="*", linewidth=1.0)
#plt.plot(VehicleSpeed,linestyle="none", marker="*", linewidth=1.0)
plt.show()

plt.close(2)
plt.figure(2)
plt.plot(LonAcc,linestyle="none", marker="*", linewidth=1.0)
#plt.plot(LatAcc,linestyle="none", marker="*", linewidth=1.0)
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

plt.close(4)
plt.figure(4)
plt.plot(VehicleSpeed,LonAcc,linestyle="none", marker="*", linewidth=1.0)
plt.show()