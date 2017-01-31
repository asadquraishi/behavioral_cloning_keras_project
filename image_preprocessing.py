# Load images into an array
from skimage.io import imread_collection
from skimage.io import concatenate_images
import numpy as np
from numpy import genfromtxt
from scipy.misc import imresize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle

print("Loading center images...")
images_center = imread_collection('track1_IMG_2/center*.jpg')
image_array_center = concatenate_images(images_center)
print("Loading left images...")
images_left = imread_collection('track1_IMG_2/left*.jpg')
image_array_left = concatenate_images(images_left)
print("Loading right images...")
images_rigth = imread_collection('track1_IMG_2/right*.jpg')
image_array_right = concatenate_images(images_rigth)
print("Finished loading images.")
image_array = np.concatenate((image_array_center,image_array_left,image_array_right))
images_right = []
images_left = []
images_center = []

# Load Steering angles into array
print("\nLoading steering angles...")
file_data = genfromtxt('track1_IMG_2_driving_log.csv', delimiter=',')
print("Finished loading steering angles.")

# Concatenate array for left, center, right images
angle = file_data[...,3]
angle = np.concatenate((angle,angle,angle))

# Create a normalizer for both images and angles
def normalizer(array, min_max=(0,1), feature_range=(0, 1)):
    x_min = feature_range[0]
    x_max = feature_range[1]
    a = min_max[0]
    b = min_max[1]
    norm_features = (array-x_min)*(b-a)/(x_max - x_min) + a
    return norm_features

# Normalize the steering angles to between -0.5 and 0.5. They are currently recorded to between -1.0 and 1.0
print("\nNormalizing steering angles...")
angle_nomalized = normalizer(angle, min_max=(-0.5,0.5), feature_range=(-1.0,1.0))
print("Steering angles normalized")

# Reduce image size to map NVidia NN
print("\nResizing images to fit NN model...")
resized_images = [imresize(image, (100,200,3))[16:,:,:][:66,:,:] for image in image_array]
resized_images = np.array(resized_images)
print("Images resized")

# Normalize the image channels to between 0 and 1
print("\nNormalizing image channels...")
normalized_images = normalizer(resized_images, min_max=(0,1), feature_range=(0,255))
print("Normalization complete")
resized_images = []

#Shuffle the data
print("\nShuffling the data")
X_train, y_train = shuffle(normalized_images, angle_nomalized)
print("Data shuffled")
normalized_images = []

# Perform a train / test split
print("\nSplit into train, validation and test data")
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=36)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=97)
print("Train size {}, validation size {}, test size {}".format(len(y_train), len(y_validation), len(y_test)))

# Save the file to a pickle so I don't have to keep processing the images
image_file = 'image_train_data.pkl'
print("\nSaving images to", image_file)
pickle_out = open(image_file, 'wb')
pickle.dump([X_train, y_train], pickle_out)
pickle_out.close()

image_file = 'image_val_data.pkl'
print("\nSaving images to", image_file)
pickle_out = open(image_file, 'wb')
pickle.dump([X_validation, y_validation], pickle_out)
pickle_out.close()

image_file = 'image_test_data.pkl'
print("\nSaving images to", image_file)
pickle_out = open(image_file, 'wb')
pickle.dump([X_test, y_test], pickle_out)
pickle_out.close()


print("\nPreprocessing complete")