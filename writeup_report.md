# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/captured_images.png "ImageCapture"
[image2]: ./examples/augmented_image.png "Augmentation"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 128 and 1 (model.py [3]) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting.
The model was trained for optimum number of epochs to reduce overfitting.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 87). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and driving in opposite direction. I also tried driving in the different track for a small distance.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

1. The overall strategy for deriving a model architecture was to collect enough driving data covering most of the driving conditions.

2. After data collection, I had 28752 data points. I split the data into training and validation data in the ratio 80:20.

3. I used lambda layer to normalize the image pixel to be between 0 and 1 and mean centered to zero. When I trained this model, training loss and validation were relatively small.
	(line 74)
4. I randomly shuffled the data set and put 20% of the data into a validation set. 

5. I implemented a proven LeNet architecture to train model expecting a good trained model. I used 2 layers of convoltion and max pooling combination with relu activation. I used a kernel 
	size of 5X5. After that I used to Dense layer to narrow down to single depth (single sterring value) over 128, 84 and 1.

6. Next I used the camera images from other 2 cameras i.e. left and right camera images. For the left and right cameras, 
I added a correction factor of 0.2 to the steering angle values (lines 24-32)
	

7. As a part of data augmentation, I flipped all the above images (center, right and left) and provided corresponding negative steering angles. At this point the vehicle was able to drive
   for a long distance but fell off track just before the bridge. (lines 48-51)

8. At last, I cropped the images to remove the distracing factors in the images like trees and the sky. I cropped 70 pixels in the top and 25 pixels in the bottom. There was a significant improvement
after this step.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

The driving simulator would save frames from three front-facing "cameras", recording data from the car's point of view from center, left anf right.
It also collects various driving statistics like throttle, speed and steering angle. We use only steering angle here.

1. Firstly, I tried to ensure car drives down the center of the road to capture center lane driving.
2. Next, Repeated this for 2-3 laps.
3. Then, drove the car in opposite direction to avoid the car always steering to the left.
4. Also, drove the car in different track which required quick changes in steering angles to train better.

The captured images from all 3 cameras is in [image1] and the augmented image is in [image2].

To augment the data sat, I also flipped images and angles thinking that this would to avoid the car pulling too hard in any one direction.

After the collection process, I had 28752 number of data points. I then preprocessed this data by using Lambda layer to normalize the image pixel to be between 0 and 1 and mean centered to zero.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5, even though initially I tried with various values like 2, 3 & 8. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
