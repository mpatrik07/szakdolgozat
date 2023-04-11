import cv2
import numpy as np
import statistics
#2.link

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth*focalLength)/perWidth

def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image*measured_distance)/real_width
    return focal_length_value

net = cv2.dnn.readNet('yolov3_training_last1.weights', 'yolov3_testing.cfg')

classes = []
tavlista = []
with open("F:\\szakdoga\\classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('F:\\szakdoga\\egyes.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

"""
kozeppontX = []
kozeppontY = []
"""
known_distance = 92.5 #eredeti távolság a kamerától
known_width = 24 #labda valós mérete cm-ben
pixel = 700/600*4120 #labda szélessége a képen pixelben
foc_length = focal_length(known_distance,known_width,pixel)

while True:
    _, img = cap.read()
    img = cv2.resize(img,(600,600))
    height, width, _ = img.shape
    a = 2048
    b = 2048
    blob = cv2.dnn.blobFromImage(img, 1/255, (a, b), (0,0,0), swapRB=True, crop=False)#a ()-nak osztható kell lenni 32-vel
    net.setInput(blob)                                                                    #egyes/kettes: (896,896), beta:(640,640)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    #xList = [item for item in range(0,a)]
    egyutthato = np.zeros(shape=(3,1))
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                tavolsag = distance_to_camera(known_width,foc_length,w)
                tavlista.append(tavolsag)

                """
                if center_x != None:
                    kozeppontX.append(center_x)
                    kozeppontY.append(center_y)
                """
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            #cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            cv2.putText(img, str(round(tavolsag,2)), (x, y+20), font, 2, (255,255,255), 2)
            if len(tavlista)>25: #(15,10,30),(25,15,50)
                print([statistics.mean(tavlista[15:]),statistics.stdev(tavlista[15:])])
            if len(tavlista)>50:
                break
            """
            if len(kozeppontX)>0:
                egyutthato = np.polyfit(kozeppontX,kozeppontY,2)
            else:
                pass
    for k in xList:
        t = int(egyutthato[0]*k**2+egyutthato[1]*k+egyutthato[2])
        cv2.circle(img,(k,t),5,(255,0,255),cv2.FILLED)
            """

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break
    
"""
vegso_egyutthato = np.polyfit(kozeppontX,kozeppontY,2)
print(vegso_egyutthato) # amikor túl van az íven, escape, és akkor kiírja
"""
cap.release()
cv2.destroyAllWindows()
