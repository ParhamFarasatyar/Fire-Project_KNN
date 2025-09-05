# Importing modules that we need
import numpy as np
from joblib import load
import cv2
import glob

# 1.Loading model in a variable
clf = load('fire_model.z')
# 2.Finding test's data location
for address in glob.glob(r'test\*'):
    img = cv2.imread(address)
    # Normalization data
    img_new = cv2.resize(img, (32, 32))
    img_new = img_new / 255.0
    img_new = img_new.flatten()
    img_new = np.array([img_new])
    
    # Prediction
    pre = clf.predict(img_new)[0]
    
    # Making text for show on images
    cv2.putText(img, str(pre), (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (3, 161, 66), 2)
    # value, text, position, font face, font scale, color(BGR), size
    cv2.imshow('Picture', img)
    cv2.waitKey(3000)