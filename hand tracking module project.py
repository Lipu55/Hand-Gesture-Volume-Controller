# -*- coding: utf-8 -*-
"""
Created on Fri May 12 21:34:02 2023

@author: MRUTYUNJAY
"""

import cv2
import mediapipe as mp
import math
import time
class handDetector():
    def __init__(self,mode=False,maxHands=1,detectionCon=1,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        self.tipIds=[4,8,12,16,20]
    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0,draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmlist=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)
            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,ymin,xmax,ymax
            
            if draw:
                cv2.rectangle(img,(bbox[0]-20,bbox[1]-20),(bbox[2]+20,bbox[3]+20),(0,255,0),2)
        return self.lmlist,bbox
    def fingerUp(self):
        finger=[]
        
        if self.lmlist[self.tipIds[0]][1]>self.lmlist[self.tipIds[0]-1][1]:
            finger.append(1)
        else:
            finger.append(0)
            
        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2]<self.lmlist[self.tipIds[id]-2][2]:
                finger.append(1)
            else:
                finger.append(0)
        return finger
    def findDistance(self,p1,p2,img,draw=True):
        x1,y1=self.lmlist[p1][1],self.lmlist[p1][2]
        x2,y2=self.lmlist[p2][1],self.lmlist[p2][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        
        if draw:
            cv2.circle(img,(x1,y1),15,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(255,0,0),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
            cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        length=math.hypot(x2 - x1, y2 - y1)
        return length,img,[x1,y1,x2,y2,cx,cy]
def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success,img=cap.read()
        img=detector.findHands(img)
        lmlist=detector.findPosition(img)
        if len(lmlist) !=0:
            print(lmlist[1])
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()