#Behavrioal Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://github.com/asadquraishi/behavioral_cloning_keras_project/blob/image_pre/nvidea-cnn-architecture.png "Model architecture"
[image2]: https://github.com/asadquraishi/behavioral_cloning_keras_project/blob/image_pre/center.jpg "Centre lane driving"
[image3]: https://github.com/asadquraishi/behavioral_cloning_keras_project/blob/image_pre/recovery_left_right.jpg "Recovery left to right"
[image4]: https://github.com/asadquraishi/behavioral_cloning_keras_project/blob/image_pre/recovery_right_left.jpg "Recovery right to left"
[image5]: https://github.com/asadquraishi/behavioral_cloning_keras_project/blob/image_pre/1_rotate_6_degrees.png "Rotated Image"
[image6]: https://github.com/asadquraishi/behavioral_cloning_keras_project/blob/image_pre/2_flip_image.png "Flipped Image"
[image7]: https://github.com/asadquraishi/behavioral_cloning_keras_project/blob/image_pre/3_crop_image.png "Cropped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network as per the [NVidia paper](https://arxiv.org/pdf/1604.07316v1.pdf) containing 5 convolutional layers of varying depth (lines 84 to 102) followed by four fully-connected layers (lines 108 to 120).

The model includes RELU layers to introduce nonlinearity (every convolutional and fully connected layer), and the data is normalized in the model using a custom normalizer (line 32).

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 89). A dropout layer before the input layer and after each of the layers was attempted. Only the remaining layer worked. I also attempted l2 normalizaion on each layer however the model didn't work with this in place.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I also tried loading an already trained model, training it on additional sets of data to see the impact of data choices on training. This is the purpose of lines 70-77.

####3. Model parameter tuning

The model used an adam optimizer, and the learning rate was tuned manually (model.py line 122).

The batch size was also experimented with. Sizes higher than 20 were not reliable so a size of 20 was chosen.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also drove multiple laps in both directions on the track.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVidia paper referenced at the beginning of this paper.

Since the NVidia architecture was chosen based on image size I ensured my input data conformed to that size. I then tried a variety of approaches including:
* adding and removing dropout layes
* adding and removing l2 regularization
* using tanh or relu activations

I used a small training set with balanced data to train the model each time a change was made including the learning rate. I made one change at a time in order to isolate the effect of the change and be able to attribute it to a specific parameter.

Data was split into train and validation sets. This allowed me to relate the scoring of the model with its behaviour on the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the layers and layer sizes shown in the image below.

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it strayed off the road.... These images show what a recovery looks like starting from left to right, then right to left:

![alt text][image3]
![alt text][image4]

After the collection process, I had approximately 30,000 data points. To augment the data sat, I created a generator pipeline that would do the following:
* rotate the image to a random angle between -10 and + 10 degrees (this number was arrived at after much experimentation.
* cropping the image so only the road and curve of the road were visible (the cropping area and amount was determined by experimentation)
* flipping one out of 4 images after rotation
* adjusting the angle beased on the images rotation

![alt text][image5]
![alt text][image6]
![alt text][image7]

The use of generators also allowed me to load data on a batch by batch basis into memory. As I added driving data, loading all of it into memory became impossible.

I finally randomly shuffled the data set and put 20% of the data into a validation set. This gave me over 7000 validation images, allowing be to attribute significance to an approximate 2% validation gain (rule of 30).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 - I made sure I terminated the training early rather than allowing it to overtrain and overfit.
