# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:58:33 2019

@author: zhuguohua
"""
####################
## 车辆减速模型分析脚本
####################
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from scipy import optimize
#import seaborn as sns 

###############################################################################
dat = np.loadtxt("../../NXP_DataSet/2019_12_3/20191203165405_Test_Data.txt")

time_err = dat[:,0]

WheelSpeedRearLeftPulseSum = dat[:,3]
WheelSpeedRearRightPulseSum = dat[:,4]

lon_acc = dat[:,6]

PulseUpdateVelocity = dat[:,9]

ActualAccelerationACC = dat[:,11]
Torque = dat[:,12]
TargetVehicleSpeed = dat[:,13]
TargetDistance = dat[:,14]

WheelSpeedMiddlePulseSum = []
for i in range(len(time_err)):
    WheelSpeedMiddlePulseSum.append( (WheelSpeedRearLeftPulseSum[i] + WheelSpeedRearRightPulseSum[i]) * 0.5)

plt.close(1)
plt.figure(1)
plt.grid(1)
plt.plot(PulseUpdateVelocity,linestyle="none", marker="*", linewidth=1.0)
plt.plot(lon_acc,linestyle="none", marker="*", linewidth=1.0)
plt.plot(ActualAccelerationACC,linestyle="none", marker="*", linewidth=1.0)
plt.plot(TargetVehicleSpeed,linestyle="none", marker="*", linewidth=1.0)
plt.plot(TargetDistance,linestyle="none", marker="*", linewidth=1.0)
plt.show()