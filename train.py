import csv
import cv2
import numpy as np
import keras
import sys
import tqdm
import random

images = []
measurements = []

# normalizes steering angle to between [-1,1]
def normalize_steering(angle):
    return angle / 25

# normalizes speed to roughly between [-1,1]
def normalize_speed(speed):
    return speed / 30 - 0.5

# adds measurements to the dataset from the given image path and steering angle/speed
def add_measurement(driving_log_dir, img_path, steering_angle, speed):

    #randomly drop 60% of the straight training data
    prob = random.random()
    if prob > 0.4 and abs(steering_angle) < 0.05 :
        return # don't include this line in the data

    filename = img_path.split('/')[-1]
    current_path = driving_log_dir + '/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # add the image
    images.append(image)
    measurements.append([normalize_steering(steering_angle), normalize_speed(speed)])

    # add a flipped version of the image
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurements.append([-normalize_steering(steering_angle), normalize_speed(speed)])


# read in the driving logs passed in as command line arguments
for driving_log_dir in sys.argv[1:]:
    with open(driving_log_dir + '/driving_log.csv') as csvfile:
        lines = [line for line in csv.reader(csvfile)]

    lines = lines[1:]

    print('loading images')
    for line in tqdm.tqdm(lines):
        # add center, left and right images to measuremnts
        add_measurement(driving_log_dir, line[0], float(line[3]), float(line[6])) #center
        add_measurement(driving_log_dir, line[1], float(line[3])+0.1, float(line[6])) #left
        add_measurement(driving_log_dir, line[2], float(line[3])-0.1, float(line[6])) #right

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# create a VGG-derived convolutional neural network.
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

model.summary()

# use an adam optimzer so that manual training of hyperparameters aren't needed
model.compile(loss='mse', optimizer='adam')

# show the training in tensorboard once per epoch
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=0,
    batch_size=32,
    write_graph=True,
    write_grads=False,
    write_images=True,
    embeddings_freq=0,
    )

# split the training data into 80% train, 20% validation
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2, callbacks=[tensorboard])

model.save('model.h5')
