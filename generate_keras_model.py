import os
import pickle
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D

# Load file from pickle
data_file = open('image_train_data.pkl', 'rb')
X_train, y_train = pickle.load(data_file)
data_file.close()

data_file = open('image_val_data.pkl', 'rb')
X_validation, y_validation = pickle.load(data_file)
data_file.close()

data_file = open('image_test_data.pkl', 'rb')
X_test, y_test = pickle.load(data_file)
data_file.close()

#Train NN
# 1st Layer - Convnet
model = Sequential()
model.add(Convolution2D(24, 5, 5,border_mode='valid',input_shape=X_train.shape[1:],subsample=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))

# 2nd Layer - Convnet
model.add(Convolution2D(36, 5, 5,border_mode='valid',subsample=(2, 2)))
model.add(Dropout(0.5))
#model.add(Activation('relu'))

# 3rd Layer - Convnet
model.add(Convolution2D(48, 5, 5,border_mode='valid',subsample=(2, 2)))
model.add(Dropout(0.5))
#model.add(Activation('relu'))

# 4th Layer - Convnet
model.add(Convolution2D(64, 3, 3,border_mode='valid'))
model.add(Dropout(0.5))
#model.add(Activation('relu'))

# 5th Layer - Convnet
model.add(Convolution2D(64, 3, 3,border_mode='valid'))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))

# Flatten
model.add(Flatten())

# Fully connected layer
model.add(Dense(100))
model.add(Dropout(0.5))
#model.add(Activation('relu'))

# Fully connected layer
model.add(Dense(50))
#model.add(Activation('relu'))

# Fully connected layer
model.add(Dense(10))
#model.add(Activation('relu'))

# Output
model.add(Dense(1))

# Train the model
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=15, nb_epoch=10, validation_split=0.0, validation_data=(X_validation, y_validation),verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save the model
json_model = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json_model)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")