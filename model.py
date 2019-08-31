#!/usr/bin/env python
# coding: utf-8

# In[9]:


import csv
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

aug_images = []
aug_angles= []
car_images = []
steering_angles = []
correction = 0.2

#Read the csv file after capturing training data.
#Read images from all 3 cameras, add/subract correction factor (0.2)
# to left/right images respectively. 
with open('drivingdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        steering_center = float(row[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        source_path_center = row[0].split('\\')[-1]
        source_path_left   = row[1].split('\\')[-1]
        source_path_right  = row[2].split('\\')[-1]
        image_center = ndimage.imread('drivingdata/IMG/' + source_path_center)
        image_left = ndimage.imread('drivingdata/IMG/' + source_path_left)
        image_right = ndimage.imread('drivingdata/IMG/' + source_path_right)
        
        # add images and angles to data set
        car_images.extend([image_center, image_left, image_right])
        steering_angles.extend([steering_center, steering_left, steering_right])

        
#f, axarr = plt.subplots(1,4)
#axarr[0].imshow(image_center)
#axarr[1].imshow(image_left)
#axarr[2].imshow(image_right)
#plt.savefig('captured_images.png')

# After all the images and their steering angles is captured,
# flip every image and apply negative of corresponding steering angle.
for image, angle in zip(car_images, steering_angles):
    image_flipped = np.fliplr(image)
    aug_images.append(image_flipped)
    measurement_flipped = -angle
    aug_angles.append(measurement_flipped)    

#plt.imshow(aug_images[0])
#plt.savefig('augmented_image.png')

car_images.extend(aug_images)
steering_angles.extend(aug_angles)

X_train = np.array(car_images)
y_train = np.array(steering_angles)
print("complete:", len(X_train), len(y_train))


# In[3]:


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, kernel_size=(5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, kernel_size=(5,5), activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')


# In[ ]:





# In[ ]:




