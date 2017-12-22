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
im = cv2.imread('./demo.jpg')
    
start = time.time()


    
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

 
cv2.imwrite('result.jpg',imwrite)
cv2.imshow('aa',imwrite)
print (time.time()-start)
cv2.waitKey(0)

    
