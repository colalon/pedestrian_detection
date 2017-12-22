import cv2
from keras.models import load_model
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
#import cnn_model
import nms
import time
import detect_people

model = load_model('none1.h5')
cap = cv2.VideoCapture('TownCentre960540all.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter( 'aa.avi',fourcc, 30.0, (960,540) )

ret = True 

while ret == True:
    
    start = time.time()
    #ret , im = cap.read()
    
    for i in range (0,1):
        ret , im = cap.read()
        rr=1
        tw = int(im.shape[1]*rr)
        th = int(im.shape[0]*rr)
        im = cv2.resize(im,(tw,th))
        
    im_gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    #loc_result = detect_people.detect_people(model,im_gray,0.75,0.5)
    loc_result = detect_people.detect_people_mutiple_size(model,im_gray,0.85,0.4,0.8,1.4,1.1)
    
    imwrite = im.copy()
    
    
    for i in range (0,loc_result.shape[0]):
        x = int(loc_result[i,0])
        y = int(loc_result[i,1])
        x2 = int(loc_result[i,2])
        y2 = int(loc_result[i,3])
        s = loc_result[i,4]
        cv2.rectangle(imwrite,(x,y),(x2,y2),(0,255,0),2)
        ss = "%.2f" % s
        cv2.putText(imwrite, str(ss), (x, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    imwrite = cv2.resize(imwrite,(960,540))   
    cv2.imwrite('result.png',imwrite)
    cv2.imshow('aa',imwrite)
    out.write(imwrite)
    cv2.waitKey(1)
    print (time.time()-start)
    
out.release()