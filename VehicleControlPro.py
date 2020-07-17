# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:51:52 2019

@author: zhuguohua
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns 

FilePath = "../../NXP_DataSet/加速度测试/"
FileName = "20200320154820_Test_Data"
dat = np.loadtxt(FilePath + FileName + ".txt")

time_err = dat[:,0]

WheelSpeedRearLeftData  = dat[:,1]
WheelSpeedRearRightData = dat[:,2]

WheelSpeedRearLeftPulse  = dat[:,3]
WheelSpeedRearRightPulse = dat[:,4]

VehicleSpeed             = dat[:,5]

LonAcc = dat[:,6]
LatAcc = dat[:,7]
YawRate = dat[:,8]

PulseUpdateVelocity = dat[:,9]
AccUpdateVelocity = dat[:,10]

ActualAccelerationACC = dat[:,11]
Torque = dat[:,12]

### control target
TargetVehicleSpeed  = dat[:,13]
TargetDistance      = dat[:,14]
### Steering Angle
SteeringAngleTarget = dat[:,15]
SteeringAngleActual = dat[:,16]
SteeringAngleSpeed  = dat[:,17]
### 跟踪位置信息
TrackPoint_Position_X = dat[:,18]
TrackPoint_Position_Y = dat[:,19]
TrackPoint_Yaw        = dat[:,20]
#### 滑模变量
TargetTrack_X   = dat[:,21]
TargetTrack_Y   = dat[:,22]
Sliding_x1      = dat[:,23]
Sliding_x2      = dat[:,24]
SlidingVariable = dat[:,25]

#x = []
#y = []
#for t in np.arange(-np.pi,np.pi,0.001):
#    r = np.sqrt(np.cos(2*t))
#    a = r*10
#    x.append(a*np.cos(t))
#    y.append(a*np.sin(t))
    
### figure
plt.close()

#plt.figure()
#plt.plot(SlidingVariable)
#plt.title('SlidingVariable')
#plt.grid(True)
#
#plt.figure()
#plt.plot(Sliding_x1,Sliding_x2)
#plt.title('Phase')
#plt.xlabel("x1")
#plt.ylabel("x2")
#plt.grid(True)

plt.figure()
plt.subplot(2,1,1)
plt.plot(TrackPoint_Position_X,SteeringAngleTarget,'--',label='target_steer')
plt.plot(TrackPoint_Position_X,SteeringAngleActual,'-',label='actual_steer')
plt.legend()
plt.title('SteeringAngle')
plt.xlabel("x[m]")
plt.ylabel("angle[deg]")
plt.subplot(2,1,2)
plt.plot(TrackPoint_Position_X,SteeringAngleSpeed,'.',label='actual_steer_rate')
plt.legend()
plt.title('SteeringAngleRate')
plt.xlabel("x[m]")
plt.ylabel("angle_rate[deg/s]")
plt.grid(True)
#plt.savefig(FilePath+FileName+"_SteeringAngle.png",dpi=200)

plt.figure()
plt.subplot(2,1,1)
plt.plot(TargetTrack_X,TargetTrack_Y,'--',label='target_track')
plt.plot(TrackPoint_Position_X,TrackPoint_Position_Y,'-',label='actual_track')
plt.grid(True)
plt.title('Tracking')
#plt.axis("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.legend()
#plt.savefig(FilePath+FileName+"_Tracking.png",dpi=200)

plt.subplot(2,1,2)
plt.plot(TargetTrack_X,TrackPoint_Position_X-TargetTrack_X,'*',label='ex')
plt.plot(TargetTrack_X,TrackPoint_Position_Y-TargetTrack_Y,'*',label='ey')
plt.grid(True)
plt.legend()
plt.title('err')
plt.xlabel("p[cnt]")
plt.ylabel("err[m]")
#plt.savefig(FilePath+FileName+"_ey.png",dpi=200)

plt.figure()
plt.plot(LonAcc,'-',label='lon')
plt.legend()
plt.savefig(FilePath+FileName+"_lon.png",dpi=200)
plt.show()