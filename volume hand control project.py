# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:25:31 2023

@author: MRUTYUNJAY
"""

import cv2
import mediapipe
import math
import time
import numpy as np
import HandTrackingmodule as htm
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#############
wCam,hCam=640,480
#############
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector=htm.handDetector(maxHands=1)

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()
minvol=volRange[0]
maxvol=volRange[1]
vol=0
volBar=400
volPer=0
area=0
colorVol=(255,0,0)
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmlist,bbox=detector.findPosition(img,draw=True)
    if len(lmlist) !=0:
        #filter based on size
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])//100
        #print(area)
        if 200<area<1000:
            #print("yes")
            # find Distance between index and thumb
            length,img,lineInfo=detector.findDistance(4,8,img)
            #print(length)
            #convert volume
            volBar=np.interp(length,[30,300],[350,150])
            volPer=np.interp(length,[50,300],[0,100])
            #Reduce resolution to make it smoother
            smoothness=10
            volPer=smoothness*round(volPer/smoothness)
            #check fingerUP
            finger=detector.fingerUp()
            print(finger)
            #if pinky is down set volume
            if not finger[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100,None)
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                colorVol=(0,255,0)
            else:
                colorVol=(255,0,0)
    # Drawing            
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
    cv2.putText(img,f'{int(volPer)}%',(40,450),cv2.FONT_ITALIC,1,(0,250,0),3)
    cVol=int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img,f'Vol set: {int(cVol)}',(400,50),cv2.FONT_ITALIC,1,colorVol,3)
    # Frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime 
    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_ITALIC,1,(255,0,0),3)
    cv2.imshow("Volume Controller",img)
    cv2.waitKey(1)