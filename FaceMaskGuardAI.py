# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:56:47 2020

@author: Karan
"""

import numpy as np
import h5py 
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime
import pyttsx3  # Added for text-to-speech

# Initialize text-to-speech engine
engine = pyttsx3.init()

# UNCOMMENT THE FOLLOWING CODE TO TRAIN THE CNN FROM SCRATCH

# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        r'I:\AVA Intern\Data Science\Data Science\Python _projects\Train',
        target_size=(150,150),
        batch_size=16 ,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        r'I:\AVA Intern\Data Science\Data Science\Python _projects\Test',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

validation_set = test_datagen.flow_from_directory(
        r'I:\AVA Intern\Data Science\Data Science\Python _projects\Validation',
        target_size=(150,150),
        batch_size=16,
        class_mode='binary')

model_saved=model.fit(
        training_set,
        epochs=10,
        validation_data=validation_set,

        )

model.save('mymodel.h5',model_saved)

#To test for individual images

mymodel=load_model('mymodel.h5')
#test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
test_image=image.load_img(r'C:\Users\vijay\Downloads\FaceMaskDetector-master\FaceMaskDetector-master\test\with_mask\1-with-mask.jpg',target_size=(150,150,3))
test_image
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
mymodel.predict(test_image)[0][0]


# IMPLEMENTING LIVE DETECTION OF FACE MASK

mymodel=load_model('mymodel.h5')

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Variable to track last announcement time
last_announce_time = datetime.datetime.now()
announce_cooldown = datetime.timedelta(seconds=3)  # 3 seconds cooldown

while cap.isOpened():
    _,img=cap.read()
    face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
    current_time = datetime.datetime.now()
    
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        test_image=image.load_img('temp.jpg',target_size=(150,150,3))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        pred=mymodel.predict(test_image)[0][0]
        if pred==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            # Announce "without mask" with cooldown
            if current_time - last_announce_time > announce_cooldown:
                engine.say("Person without mask detected")
                engine.runAndWait()
                last_announce_time = current_time
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            # Announce "with mask" with cooldown
            if current_time - last_announce_time > announce_cooldown:
                engine.say("Person with mask detected")
                engine.runAndWait()
                last_announce_time = current_time
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
          
    cv2.imshow('img',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()