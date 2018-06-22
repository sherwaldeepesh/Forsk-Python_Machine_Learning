# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:50:32 2018

@author: deepe
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):
    #"Frame" will get the next frame in the camera (via "cap").
    #"ret" will obtain return value from getting the camera frame, either true of false.
    ret,frame = cap.read()
    #we use the function cv2.cvtColor(input_image, flag) where flag determines the type of conversion.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #defining the range of red color
    red_lower = np.array([136,87,111],np.uint8)
    red_upper = np.array([180,255,255],np.uint8)

    #defining the range of blue color
    blue_lower = np.array([99, 115,150],np.uint8)
    blue_upper = np.array([110,255,255],np.uint8)

    #defining the range of yellow color
    yellow_lower = np.array([22,60,200],np.uint8)
    yellow_upper = np.array([60,255,255],np.uint8)

    #finding the range of red, blue and yellow color in the image
    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    #Morphological transformation, Dilation
    #The kernel slides through the image (as in 2D convolution).
    #A pixel in the original image (either 1 or 0) will be considered 1
    #only if atleast one pixel under the kernel is '1'.
    kernal = np.ones((5,5), "uint8")
    #print kernal

    red = cv2.dilate(red, kernal)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame,mask= red)

    blue = cv2.dilate(blue, kernal)
     # Bitwise-AND mask and original image
    res1 = cv2.bitwise_and(frame,frame,mask= blue)

    yellow = cv2.dilate(yellow, kernal)
     # Bitwise-AND mask and original image
    res2 = cv2.bitwise_and(frame,frame,mask= yellow)

    #Tracking the Red color
    
    (ret,contours,hierarchy) = cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>300:
            x,y,w,h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))


    #Tracking the Blue color
    (ret,contours,hierarchy) = cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>300:
            x,y,w,h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,"BLUE color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0))

    #Tracking the Green color
    (ret,contours,hierarchy) = cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>300:
            x,y,w,h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,"GREEN color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0))

    cv2.imshow("Color Tracking",frame)
    if cv2.waitKey(10)& 0xff == ord('S'):
        cap.release()
        cv2.destroyAllWindows()
        break
    