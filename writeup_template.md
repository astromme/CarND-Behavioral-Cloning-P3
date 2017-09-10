**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[centerImage]: ./examples/center_2017_08_30_19_59_56_933.jpg "Center Image"
[leftImage]: ./examples/left_2017_08_30_20_02_54_619.jpg "Left Image"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* successful_model.h5 containing a trained convolution neural network
* writeup_template.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py successful_model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a VGG-derived convolution neural network with 3x3 filter sizes and depths between 32 and 256.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only used
center lane driving and simulated recovery rather than capturing it directly.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I had good luck with a VGG-derived model architecture for the traffic signs
project, so I started with the same architecture for this project. I was able
to achieve good results without further changes to this model. Instead, I
added preprocessing steps and generated additional training data.

My VGG-derived model has 4 groups of stacked 3x3 convolutions with two dense
layers at the end. You can see the model code here:

```
model = Sequential()
# 50 from top, 20 from bottom, 0, 0 from left and right
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))
```

I first started with simply training on the full center images, but this
did not perform well with the car driving off the track frequently. I then
augmented the training set with the left and right off-center images with a
steering adjustment of +- 1 degree, which performed better at staying on track
but tended to weave back and forth between the far edges of the lane like
an underdamped spring.

I attempted to fix this by decreasing the steering adjustment and by adding
additional training data by recording my own steering around the track. I also
augmented the dataset by flipping left-right and reduced the size by cropping
away the top and bottom sky and car. This improved performance, but didn't
fully solve the steering issue.

What finally solved the underdamped steering was to reduce the amount of
training data that showed the car simply steering straight by randomly throwing
away 60% of the samples where magnitude of the steering angle was less than
0.05. This made the dataset more balanced, and drastically reduced the
underdamped steering I had observed.

My vehicle is able to drive laps around the lake without ever leaving the road.
I was even able to increase the target speed from 9mph to 20mph and still
have the car stay on the road.

I also experimented with letting the car set its own target speed. In theory
this should allow it to slow down around turns. 

#### 3. Creation of the Training Set & Training Process

I found that I didn't need recovery data, and that using the left & right images
with steering adjustment was sufficient. My training data consisted entirely of
center lane data.

This is an example of a center image:

![center image][centerImage]

This is an example of a generated "recovery" left image:

![left image][leftImage]
