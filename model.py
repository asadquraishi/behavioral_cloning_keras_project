from skimage.io import imread
import numpy as np
from skimage.transform import rotate
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas import read_csv
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from math import ceil

nb_epoch = 5
batch_size = 20
rotation_angle = 10
image_dir = 'IMG'

'''
# Load file from pickle
data_file = open('image_train_data.pkl', 'rb')
X_train, y_train = pickle.load(data_file)
data_file.close()

data_file = open('image_val_data.pkl', 'rb')
X_validation, y_validation = pickle.load(data_file)
data_file.close()

data_file = open('image_test_data.pkl', 'rb')
X_test, y_test = pickle.load(data_file)
data_file.close()'''

def load_data():
    col_names = ['centre', 'left', 'right', 'angle', 'throttle', 'brake', 'speed']
    data = read_csv('driving_log.csv',header=None,names=col_names)
    centre_filename = data.centre.tolist()
    angle = data.angle.tolist()
    X_train, X_test, y_train, y_test = train_test_split(centre_filename, angle, test_size=0.2, random_state=36)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=36)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Create a normalizer for both images and angles
def normalizer(array, min_max=(0,1), feature_range=(0, 1)):
    x_min = feature_range[0]
    x_max = feature_range[1]
    a = min_max[0]
    b = min_max[1]
    norm_features = (array-x_min)*(b-a)/(x_max - x_min) + a
    return norm_features


def data_generator(batch_size, images, angles, rotation_angle):
    images, angles = shuffle(images, angles)

    while True:
        X_data, y_data = [], []
        for index in range(batch_size):
            data_choice = np.random.randint(len(images))
            path = image_dir + '/' + images[data_choice].split('/')[-1]
            image = imread(path)
            angle = angles[data_choice]
            # crop image
            image = image[60:, :, :][:66, :200, :]
            #print('image {}, angle {}'.format(image[0,0],angle))
            # rotate image by a random angle
            rotate_by = np.random.randint(-rotation_angle, rotation_angle)
            image = rotate(image, rotate_by)
            # normalize the image and angle
            image = normalizer(image, min_max=(0, 1), feature_range=(0, 255))
            angle = normalizer(angle, min_max=(-0.5, 0.5), feature_range=(-1.0, 1.0))
            # randomly flip image
            '''if np.random.randint(2) == 1:
                image = np.fliplr(image)
                angle = -angle'''
            # add data to the array
            X_data.append(image)
            y_data.append(angle)
        #print('shape of array',np.asarray(X_data).shape)
        #print('image pixel value = {} and angle = {}'.format(X_data[0,0,0], y_data[0]))
        yield np.asarray(X_data), y_data

def build_model():

    # Load the model if we want to train it on additional data
    try:
        # Load the model if we want to train it on additional data
        with open('model.json', 'r') as jfile:
            model = model_from_json(jfile.read())

        adam = Adam(lr=0.0005)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
        model.load_weights('model.h5')
        nb_epoch = 2

        print("imported existing model")

    except:
        # Else build the model
        # 1st Layer - Convnet
        print("Build a new model")
        model = Sequential()
        #model.add(Convolution2D(24, 5, 5,border_mode='valid',input_shape=X_train.shape[1:],subsample=(2, 2)))
        model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(66,200,3), subsample=(2, 2)))
        #model.add(Dropout(0.25)) # went right
        model.add(Activation('relu'))

        # 2nd Layer - Convnet
        model.add(Convolution2D(36, 5, 5,border_mode='valid',subsample=(2, 2)))
        model.add(Dropout(0.25)) # went left
        #model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # 3rd Layer - Convnet
        model.add(Convolution2D(48, 5, 5,border_mode='valid',subsample=(2, 2)))
        #model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # 4th Layer - Convnet
        model.add(Convolution2D(64, 3, 3,border_mode='valid'))
        #model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # 5th Layer - Convnet
        model.add(Convolution2D(64, 3, 3,border_mode='valid'))
        #model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # Flatten
        model.add(Flatten())

        # Fully connected layer
        model.add(Dense(100))
        #model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # Fully connected layer
        model.add(Dense(50))
        #model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # Fully connected layer
        model.add(Dense(10))
        #model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # Output
        model.add(Dense(1))

        adam = Adam(lr=0.0001)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # load the data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print("{} train samples, {} validation samples, {} test samples loaded.".format(len(X_train), len(X_val), len(X_test)))

    # build the model
    model = build_model()
    print(model.summary())

    # Train the model
    train_generator = data_generator(batch_size=batch_size, images=X_train, angles=y_train,
                                     rotation_angle=rotation_angle)
    val_generator = data_generator(batch_size=batch_size, images=X_val, angles=y_val,
                                   rotation_angle=rotation_angle)

    history = model.fit_generator(train_generator, samples_per_epoch=ceil(len(y_train)/batch_size)*batch_size,
                                  nb_epoch=nb_epoch, validation_data=val_generator, nb_val_samples=ceil(len(y_val)/batch_size)*batch_size, verbose=1)

    #assess = model.evaluate(X_test, y_test, verbose=0)
    #print('Loss:', assess[0])
    #print('Accuracy:', assess[1])

    # Save the model
    json_model = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(json_model)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
