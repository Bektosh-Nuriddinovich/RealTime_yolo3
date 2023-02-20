import cv2
import numpy as np

cap = cv2.VideoCapture('a.mp4')
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
classFile = 'coco.names'
classNames = []

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(len(classNames))

modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    count = 0
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox)) ===> if it prints 1, it means that it found one object in the image  
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        if classNames[classIds[i]] == 'car' or classNames[classIds[i]] == 'bus' or classNames[classIds[i]] == 'truck' or classNames[classIds[i]] == 'motorbike':
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x-50, y-50), (x+w+50, y+h+50), (0,0,255), 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', 
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            #imgCar = img[y:y+h-50, x:x+w-50]
            #cv2.imwrite('CARS/CAR_'+str(count)+'.jpg', imgCar)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    #print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    outputs = net.forward(outputNames)
    #print(len(outputs))
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    findObjects(outputs, img)

    cv2.imshow('WebCam', img)
    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break