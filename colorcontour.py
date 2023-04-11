import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import statistics

#https://github.com/askitlouder/Image-Processing-Tutorials/blob/main/41%20-%20Object%20Tracking%20using%20Cam%20and%20Mean%20shift%20OpenCV.py
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth*focalLength)/perWidth

def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image*measured_distance)/real_width
    return focal_length_value

known_distance = 95 #eredeti távolság a kamerától cm-ben
known_width = 22.25 #labda valós mérete cm-ben
pixel = 700*600/4160 #labda szélessége a képen pixelben, eredetileg 700
foc_length = focal_length(known_distance,known_width,pixel)
font = cv2.FONT_HERSHEY_PLAIN
#binding trackbar with video

cap = cv2.VideoCapture('F:\\Szakdoga\\egy.mp4')#a kettokre off
def nothing(x):
    pass
mycolorfinder = ColorFinder(False)
szin = {'hmin': 16, 'smin': 137, 'vmin': 100, 'hmax': 255, 'smax': 255, 'vmax': 255} #(16,137,75), a többi 255
cv2.namedWindow("Color Adjustments")
cv2.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)
kozeppont = []
kozeppontX = []
kozeppontY = []
tavlista = []
egyutthato = []
xList = [item for item in range(0,3840)]
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(600,600))#enélkül kisebb a torzítás, de nem látható
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("Lower_H", "Color Adjustments")
    l_s = cv2.getTrackbarPos("Lower_S", "Color Adjustments")
    l_v = cv2.getTrackbarPos("Lower_V", "Color Adjustments")
    u_h = cv2.getTrackbarPos("Upper_H", "Color Adjustments")
    u_s = cv2.getTrackbarPos("Upper_S", "Color Adjustments")
    u_v = cv2.getTrackbarPos("Upper_V", "Color Adjustments")
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    #print(lower_bound)
    #print(upper_bound)
    #lower_bound = np.array([0, 137, 56])
    #upper_bound = np.array([57, 255, 255])
    mask1 = cv2.inRange(hsv, lower_bound, upper_bound)
    res = cv2.bitwise_and(frame, frame, mask=mask1)
    framecolor, mask2 = mycolorfinder.update(frame,szin)
    framecontours, contours = cvzone.findContours(frame,mask2,minArea=200)
    if contours:
        tavolsag = distance_to_camera(known_width,foc_length,contours[0]["bbox"][2])
        #print(contours)
        tavlista.append(tavolsag)
        kozeppont.append(contours[0]["center"])

    """if kozeppont:
        for i in kozeppont:
            kozeppontX.append(i[0])
            kozeppontY.append(i[1])
            cv2.circle(framecontours,i,2,(0,255,0),cv2.FILLED)
            egyutthato = np.polyfit(kozeppontX,kozeppontY,2)
            #print(egyutthato)
        for k in xList:
            t = int(egyutthato[0]*k**2+egyutthato[1]*k+egyutthato[2])
            cv2.circle(framecontours,(k,t),3,(255,0,255),cv2.FILLED)"""

    #cv2.imshow("Original Frame", frame)
    #cv2.imshow("Masking", mask1)
    #cv2.imshow("Result", res)
    cv2.imshow("Imagecontour", framecontours)

    """if len(tavlista)>2 and len(tavlista)<50:
        print([statistics.mean(tavlista),statistics.stdev(tavlista)])
    elif len(tavlista)>51:
        break"""

    if len(tavlista)>25: #(15,10,30),(25,15,50)
        print([statistics.mean(tavlista[15:]),statistics.stdev(tavlista[15:])])
    if len(tavlista)>50:
        break
    
    key = cv2.waitKey(50)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()