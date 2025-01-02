import numpy
import cv2
from pathlib import Path
from ultralytics import YOLO
import os

# tuy chinh thong so
link_data = './test'
link_model = "./best.pt"#link model
model = YOLO(link_model)

for i in os.listdir(link_data):
    for j in os.listdir(link_data+"/"+i):
        img_ = cv2.imread(link_data+"/"+i+"/"+j)
        detect_ob = model(img_,conf=0.5)
        height,width,_ = img_.shape
        xyxy = detect_ob[0].boxes.xyxy
        xywh = detect_ob[0].boxes.xywh
        class__ = detect_ob[0].boxes.cls
        if  class__ is None:
            continue
        img = img_.copy()
        for index__,k in enumerate(xywh):
                vitri = (int(xyxy[index__][0]),int(xyxy[index__][1]))
                text = ''
                if class__[index__]==0:
                    text = "not fall"
                if class__[index__]==1:
                    text = "fall"
                img = cv2.putText(img,text,vitri,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1,cv2.LINE_AA)
                img = cv2.rectangle(img,(int(xyxy[index__][0]),int(xyxy[index__][1])),(int(xyxy[index__][2]),int(xyxy[index__][3])), (0,0,255),thickness=2)
                img = cv2.circle(img,(int(k[0]),int(k[1])), 2, (223, 3, 252), 2)
        img = cv2.resize(img,(400,600))
        cv2.imshow('test',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # a = input("enter : ")
        print("________________________")
