# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model

from keras.datasets import cifar10
import numpy as np
classifier = load_model('dog_cat.h5')
model = load_model('keras_cifar10_trained_model.h5')
# Initialising the CNN
##(x_train, y_train), (x_test, y_test) = cifar10.load_data()
##print('x_train shape:', x_train.shape)
##print(x_train.shape[0], 'train samples')
##print(x_test.shape[0], 'test samples')
##
##print(x_train.shape, y_train.shape)
##print(x_test.shape, y_test.shape)
##print(type(x_test))
##print(type(y_test[0]))
### Part 3 - Making new predictions
##cifar10_labels = np.array([
##    'airplane',
##    'automobile',
##    'bird',
##    'cat',
##    'deer',
##    'dog',
##    'frog',
##    'horse',
##    'ship',
##    'truck'])


from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

#def convertCIFER10Data(image):
#    img = image.astype('float32')
#    img /= 255
#    c = np.zeros(32*32*3).reshape((1,32,32,3))
#    c[0] = img
#    return c
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

##test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (32, 32))
##test_image = image.img_to_array(test_image)
##test_image = np.expand_dims(test_image, axis = 0)
##
##result2 = model.predict(test_image)
##print(result2[0][8])
##for n in range(1,10):
##    if(result2[0][n]==1):
##        prediction2=l[n]
