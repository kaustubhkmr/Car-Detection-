# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:56:13 2017

@author: RyoKMR
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import os
import time
start = time.time()


cap = cv2.VideoCapture("input_video.3gp")

fps = cap.get(cv2.CAP_PROP_FPS)
print (fps)
try:
    if not os.path.exists('Frame'):
        os.makedirs('Frame')
        
except OSError:
    print ('Error: Creating data')
    
count = 0
success, frame = cap.read()
os.chdir('Frame')
while(success and count < 700):
    success, frame = cap.read()
    name = str(count) + '.jpg'
    cv2.imwrite(name,frame)
    count += 1
    
print ('\ntime taken = ' + str(time.time() - start))

cap.release()
cv2.destroyAllWindows()
