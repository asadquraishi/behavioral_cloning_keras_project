# Behavioral Cloning Project

##Fist exercise: Use the Centre Image

Use Keras to train a network to do the following:

1. Take in an image from the center camera of the car. This is the input to your neural network.
2. Output a new steering angle for the car.

You don’t have to worry about the throttle for this project, that will be set for you.

[Save your model](https://keras.io/models/about-keras-models/) architecture as model.json, and the weights as model.h5.

###Validating Your Network
You can validate your model by launching the simulator and entering autonomous mode.

The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

1. Set up your development environment with the CarND Starter Kit.
2. Run the server.
* python drive.py model.json
* If you're using Docker for this project: docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starer-kit python drive.py model.json or docker run -it --rm -p 4567:4567 -v ${pwd}:/src udacity/carnd-term1-starer-kit python drive.py model.json. Port 4567 is used by the simulator to communicate.

Once the model is up and running in drive.py, you should see the car move around (and hopefully not off) the track!
