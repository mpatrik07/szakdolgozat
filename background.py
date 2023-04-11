import numpy as np
import cv2
import cvzone
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import csv
#forrás: lehet a colortrackes

def nothing(x):
    pass

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth*focalLength)/perWidth

def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image*measured_distance)/real_width
    return focal_length_value

def kiirat(lista):
    with open('data.csv','w') as out:
        csv_out=csv.writer(out)
        for row in lista:
            csv_out.writerow(row)

known_distance = 95 #eredeti távolság a kamerától cm-ben
known_width = 22.25 #labda valós mérete cm-ben
pixel = 700*600/4160 #labda szélessége a képen pixelben, eredetileg 700
foc_length = focal_length(known_distance,known_width,pixel)
font = cv2.FONT_HERSHEY_PLAIN
#binding trackbar with video

cap = cv2.VideoCapture('F:\\Szakdoga\\egy.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(14,14))
fgbg = cv2.createBackgroundSubtractorMOG2()
kozeppont = []
kozeppontX = []
kozeppontY = []
tavlista = []
egyutthato = []
xList = [item for item in range(0,3840)]
gorbedata = []
while True:
    ret, frame = cap.read()
    if ret == False:
        kiirat(gorbedata)
        break
    frame = cv2.resize(frame,(600,600))
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    result2 = cv2.bitwise_and(frame, frame, mask=fgmask)
    #cv2.imshow('elso szures sargara', result2)
    #cv2.imshow('elso szures sargara', fgmask)

    mask2 = fgmask
    framecontours, contours = cvzone.findContours(frame,mask2,minArea=300)
    if contours:
        tavolsag = distance_to_camera(known_width,foc_length,contours[0]["bbox"][2])
        #print(contours)
        tavlista.append(tavolsag)
        kozeppont.append(contours[0]["center"])
        gorbedata.append((contours[0]["center"][0],600-contours[0]["center"][1],tavolsag))
    if kozeppont:
        for i in kozeppont:
            kozeppontX.append(i[0])
            kozeppontY.append(i[1])
            #cv2.circle(framecontours,i,2,(0,255,0),cv2.FILLED)
            #egyutthato = np.polyfit(kozeppontX,kozeppontY,2)
            #print(egyutthato)
        """for k in xList:
            t = int(egyutthato[0]*k**2+egyutthato[1]*k+egyutthato[2])
            cv2.circle(framecontours,(k,t),3,(255,0,255),cv2.FILLED)"""
    cv2.imshow("Imagecontour", framecontours)
    """if len(tavlista)>25: #(15,10,30),(25,15,50)
        print([statistics.mean(tavlista[15:]),statistics.stdev(tavlista[15:])])
    if len(tavlista)>50:
        break"""
    key = cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()
