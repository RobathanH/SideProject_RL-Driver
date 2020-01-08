import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import os
import random

from keras.layers import Lambda, Input, Dense, Activation
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import backend as K


ORIG_IMG_SHAPE = (80, 80)
ORIG_DIMENSIONS = ORIG_IMG_SHAPE[0] * ORIG_IMG_SHAPE[1]
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 150

# model sizes
INTERMEDIATE_LAYER_SIZE = 100
LATENT_LAYER_SIZE = 20

# training vars
EPOCH_NUM = 300
BATCH_SIZE = 64

# assumes folders contain only images
IMG_FOLDER_NAMES = [
    "C:\\Users\\Rob\\Desktop\\donkeycar\\rl\\images_crossingtrack",
    "C:\\Users\\Rob\\Desktop\\donkeycar\\rl\\images_edgetrack_bothways",
    "C:\\Users\\Rob\\Desktop\\donkeycar\\rl\\images_ontrack_bothways"
    ]



def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




# gather all images into trainable format
# take equal number of images from each folder, to avoid skewing the data

# find minimum number of images in a folder
minLen = None
for folderName in IMG_FOLDER_NAMES:
    length = len(os.listdir(folderName))
    if minLen == None or length < minLen:
        minLen = length
        
images = []
for folderName in IMG_FOLDER_NAMES:
    for imagePath in os.listdir(folderName)[:minLen]:
        img = cv2.imread(folderName + "\\" + imagePath)
        if not img is None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
            img = cv2.resize(img, ORIG_IMG_SHAPE)
            img = np.array(img)
##            cv2.imshow("test", img)
##            cv2.waitKey(1)
            img = img.flatten()
            images.append(img)

images = np.array(images)

# scale to 0-1
maxNum = np.amax(images)
images = np.divide(images, maxNum)

# split into train test
random.shuffle(images)
test_data = images[:int(0.1*len(images))]
train_data = images[int(0.1*len(images)):]
            

# encoder model
inputs = Input(shape=(ORIG_DIMENSIONS,), name='encoder_input')
x = Dense(INTERMEDIATE_LAYER_SIZE, activation='relu')(inputs)
latent_mean = Dense(LATENT_LAYER_SIZE, name='latent_mean')(x)
latent_log_var = Dense(LATENT_LAYER_SIZE, name='latent_log_var')(x)

latent_sample = Lambda(sampling, output_shape=(LATENT_LAYER_SIZE,), name='sample')([latent_mean, latent_log_var])

encoder = Model(inputs, [latent_mean, latent_log_var, latent_sample], name='encoder')

# decoder model
latent_inputs = Input(shape=(LATENT_LAYER_SIZE,), name='latent_sample')
scaled_input = Activation('sigmoid', name='scaled_sample')(latent_inputs)
x = Dense(INTERMEDIATE_LAYER_SIZE, activation='relu', name='decoder_intermediate')(scaled_input)
outputs = Dense(ORIG_DIMENSIONS, activation='sigmoid', name='decoder_output')(x)

decoder = Model(latent_inputs, outputs, name='decoder')

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

encoder.summary()
decoder.summary()

# loss
#reconstruction_loss = mse(inputs, outputs)
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= ORIG_IMG_SHAPE[0] * ORIG_IMG_SHAPE[1]
k1_loss = 1 + latent_log_var - K.square(latent_mean) - K.exp(latent_log_var)
k1_loss = K.sum(k1_loss, axis=-1)
k1_loss *= -0.5
loss = K.mean(reconstruction_loss + k1_loss)
vae.add_loss(loss)
vae.compile(optimizer='Adam')

vae.summary()

# encoder.load_weights('encoder_weights.h5')
# decoder.load_weights('decoder_weights.h5')

vae.fit(train_data, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, validation_data=(test_data, None))
encoder.save_weights('encoder_weights.h5')
decoder.save_weights('decoder_weights.h5')



# plot results for some test data
for img in test_data[:3]:
    formatted = img.reshape(ORIG_IMG_SHAPE)
    plt.imshow(formatted)
    plt.show()

    result = vae.predict([[img]])[0]
    result = result.reshape(ORIG_IMG_SHAPE)
    plt.imshow(result)
    plt.show()
