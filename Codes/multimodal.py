#
#   @author      : SRvSaha
#   Filename     : multimodal.py
#   Timestamp    : 12:12 04-November-2017 (Saturday)
#   Email        : contact [dot] srvsaha [at] gmail.com

import sys
import json
import embedding
import numpy as np
# For the input shape, for taking the input
from keras.layers import Input
# This is to convert the 2048 dimension image to 256 dimension
from keras.layers import Dense
# For the text
from keras.layers import GRU
# For creating the embedding
from keras.layers import Embedding
from keras.utils.layer_utils import merge
from keras.models import Model
from text_preprocessing import load_features_images_captions, preprocess_text
from keras import backend as K

# --------------------------------------------------------------------------- #


def loss_func(y_true, y_pred):
    # Since we have set the output of our model to be a pair of positive and
    # negative that are concatenated, so the first column contains the result
    # of dot product of original caption and image and the second column contains
    # the result of the dot product of noise(false) caption

    # Positive will be a vector(tensor) of all the training rows and 1st column
    positive = y_pred[:, 0]
    # Negative will be a vector(tensor) of all the training rows and 2nd column
    negative = y_pred[:, 1]
    # The loss function sums up all the max of (0, 1- p + n) and returns that
    return K.sum(K.maximum(0., 1. - positive - negative))


def accuracy_(y_true, y_pred):
    # Our accuracy is defined as the number of times a positive pair effectively
    # gets a higher value than a negative pair
    # Positive will be a vector(tensor) of all the training rows and 1st column
    positive = y_pred[:, 0]
    # Negative will be a vector(tensor) of all the training rows and 2nd column
    negative = y_pred[:, 1]
    # Returns the mean value (average number of times) when positive is greater
    # than negative
    return K.mean(positive > negative)


# --------------------------------------------------------------------------- #

# INPUTS
input_image = Input(shape=(2048,))
input_actual_caption = Input(shape=(16,))
input_false_caption = Input(shape=(16, ))

# --------------------------------------------------------------------------- #

# LOADING EMBEDDINGS <WEIGHTS>

# TODO : Use Argument Parser for using flags for input
# sys.argv[1] : vocabulary.json
# sys.argv[2] : dimension of output text embedding
# sys.argv[3] : Filename of the Original Glove Embedding

with open(sys.argv[1]) as f:
    # This is used to load the json file and convert it to Python Dictionary
    vocabulary = json.load(f)

embedding_weights = embedding.load(vocabulary, sys.argv[2], sys.argv[3])

# --------------------------------------------------------------------------- #

# CREATING THE LAYERS OF MODEL

# The Input Sequences are of dimension 16 and we are using the weights of 100
# dimension using GloVe that are used as initial weights
# Input Sentences are of length 16 which are shown in 16 dimension.
# Basically we are going to train our input data using GRU where the initial
# weights are initialized from GloVe rather than being
caption_embedding = Embedding(
    len(vocabulary), 100, input_length=16, weights=[embedding_weights])
# GRU is used to create the generate word embeddings of 256 dimension
caption_embedding_using_rnn = GRU(256)
# Dense Layer is used to convert the image from 2048 dimension to 256 dimension
# tanh activation is used to limit the magnitude from -1 to 1 for each vector
image_embedding_using_dense_layer = Dense(256, activation='tanh')

# --------------------------------------------------------------------------- #

# PASSING INPUT TO LAYERS FOR COMPUTING

# Since we are using the Functional API of Keras, so we can call
# or pass parameters to any layers
#
# Passing the input image to embedding dense layer
image_pipeline = image_embedding_using_dense_layer(input_image)
# The weights are shared for the noise caption(false caption) and the actual
# caption(original caption)

# First we pass the input original caption to caption_embedding which is then
# passed to GRU layer the get the representation in 256 dimension. The same is
# done for false caption as well (noise input)
actual_caption_pipeline = caption_embedding_using_rnn(
    caption_embedding(input_actual_caption))
false_caption_pipeline = caption_embedding_using_rnn(
    caption_embedding(input_actual_caption))

# --------------------------------------------------------------------------- #

# UTILITIES FOR TRAINING THE MODEL

# Basically a good matching caption to an image will have high value for
# the positive_pair and low value for the negative_pair

# Dot product of the image vector and the actual caption vector

# TODO: Make it compatible for Keras 2.x.x.
# Difference: merge in keras 1.x.x returns a scalar after dot product whereas
# it returns a tensor is 2.x.x Figure out a way to convert the tensor to scalar
# KERAS 2 SPECIFIC: different way to use dot, concatenate
positive_pair = merge([image_pipeline, actual_caption_pipeline], mode='dot')
# Dot product of the image vector with the false(noise) caption vector

negative_pair = merge([image_pipeline, false_caption_pipeline], mode='dot')

# Since we want out model to output the value of both positive pair and
# negative_pair so we concatenate them together and make a single vector.
# This will help us out in the loss function when we are training our system.

output = merge([positive_pair, negative_pair], mode='concat')

# --------------------------------------------------------------------------- #

# DEFINING ALL MODELS

# This is the main training model that is used for training the system based on
# the inputs

# TODO: Convert to Keras 2.x.x
# KERAS 2 SPECIFIC - inputs, outputs instead of input, output

training_model = Model(
    input=[input_image, input_actual_caption, input_false_caption], output=output)

# This is the image model that is used only for images for creating the dense
# representation of any image in 256 dimension

image_model = Model(input=input_image, output=image_pipeline)

# This is the caption model that will take a text as input and then convert it
# into 256 dimension using pretrained GloVe embedding and  passing it in GRU

caption_model = Model(input=input_actual_caption,
                      output=actual_caption_pipeline)

# --------------------------------------------------------------------------- #

# COMPILING THE TRAINING MODEL

# To compile the training model, we use our custom loss function and our custom
# accuracy metric along with the adam optimizer.
training_model.compile(
    loss=loss_func, optimizer='adam', metrics=[accuracy_])

# --------------------------------------------------------------------------- #

# LOADING DATA FOR TRAINING OF SYSTEM

# sys.argv[4] : captions_filename - annotations.10k.txt
# sys.argv[5] : resnet50-features.10k.npy

# Returns image features (shape : 100000, 2048), image ids, texts (before
# converting to integer sequences)
features, images, texts = load_features_images_captions(
    sys.argv[4], sys.argv[5])

# captions is a numpy array of shape (10000, 16) which consists of all texts
# padded after being converted to integer sequences
captions = preprocess_text(texts)
# Since noise are also captions but mismatched ones, so we copy the captions to
# noise with the help of numpy
noise = np.copy(captions)
# We create fake_labels which are all zero. This is required for training the
# model. But while training, this is of no use since our loss function is
# dependant only on the y_predicted not on y_train. These fake labels are for
# y_train
fake_labels = np.zeros((len(features), 1))

# --------------------------------------------------------------------------- #

# TRAIN-TEST SPLIT

# We are using first 9000 rows as TRAIN DATA and the remaining 1000 rows for
# TESTING the accuracy of our system

# Creating the training set
X_train = [features[:9000], captions[:9000], noise[:9000]]
Y_train = fake_labels[:9000]

# Creating the test/validation set: We call it validation set since we know the
# caption corresponding to each image
X_test = [features[9000:], captions[9000:], noise[9000:]]
Y_test = fake_labels[9000:]

# --------------------------------------------------------------------------- #

# SYSTEM TRAINING STARTS HERE

for epoch in range(1):
    # We need to shuffle it in every epoch so that our system doesn't start
    # learning the features of mismatched caption.
    np.random.shuffle(noise)
    # Training the model with batch size of 64 samples i.e, weights will get
    # updated per 64 samples and number of times the whole training set will be
    # iterated is 15. Alongside training, we check the output on out Validation/
    # Test set checking the accuracy
    training_model.fit(X_train, Y_train, validation_data=[
                       X_test, Y_test], nb_epoch=1, batch_size=64)
