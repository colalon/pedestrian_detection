import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import nms


def detect_people(model,gray_image,peopleThresh,overlapThresh):
    im_gray = gray_image/255
    h,w = gray_image.shape[0:2] 
    jud = im_gray.reshape((1,h,w,1))
    output = model.predict(jud)
    #out = output.copy()
    output = output.reshape((output.shape[1],output.shape[2]))
    people_loc = []
    for i in range (0,output.shape[0]):
        for j in range (0,output.shape[1]):
            if output[i,j] > peopleThresh:
                x=j*8
                y=i*8
                people_loc.append([x,y,x+64,y+128,output[i,j]])
    
    people_loc = np.asarray(people_loc)
    loc_result = nms.non_max_suppression_slow(people_loc,overlapThresh)
    
    return loc_result

def detect_people_mutiple_size(model,gray_image,peopleThresh,overlapThresh,minsize,maxsize,rate):
    im_gray = gray_image
    h,w = gray_image.shape[0:2] 
    hmin = round(h*minsize)
    wmin = round(w*minsize)
    hmax = round(h*maxsize)
    wmax = round(w*maxsize)

    dw=wmin
    dh=hmin
    
    all_loc=[]
    
    while dh < hmax or dw < wmax:
        jug_im = cv2.resize(im_gray,(dw,dh))
        loc = detect_people(model,jug_im,peopleThresh,overlapThresh)
        #print (loc==[])
        #print (loc)
        if loc != [] :
            loc[:,0:4] = loc[:,0:4]*(w/dw)
            all_loc.append(loc)
        dh = round(dh*rate)
        dw = round(dw*rate)
    all_loc = np.asarray(all_loc)
    
    size = 0
    for i in range (0,len(all_loc)):
        size = size + all_loc[i].shape[0]

    
    all_list = np.zeros((size,5))
    
    index=0
    
    for i in range (0,len(all_loc)):
        #print ('aa',all_loc[i].shape[0])
        all_list[index:index+all_loc[i].shape[0]]=all_loc[i].copy()
        index = index+all_loc[i].shape[0]
        
    loc_result = nms.non_max_suppression_slow(all_list,overlapThresh)
    
    return loc_result
        
    
        