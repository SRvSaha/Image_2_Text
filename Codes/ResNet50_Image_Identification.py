#
#   @author      : SRvSaha
#   Filename     : ResNet50_Image_Identification.py
#   Timestamp    : 12:22 25-October-2017 (Saturday)
#   Email        : contact [dot] srvsaha [at] gmail.com


# coding: utf-8

# In[4]:

# Imports
from keras.applications import ResNet50  # 50 layer Neural Network
# Helper function that uses the Imagenet Utilities
from keras.applications import imagenet_utils
from keras.preprocessing import image  # For dealing with images
import numpy as np  # Numpy array operations handling
import cv2
import sys


# In[2]:

# Dimension of Image is 224*224 for the ConvNN
inputShape = (224, 224)


# In[3]:

# Instantiating the ResNet50 NN using the activations of imagenet.
# NOTE: If the activations are not available, they'll be downloaded and will take time
model = ResNet50(weights="imagenet")


# In[4]:

def preprocessing(image_path):
    # Loading and image and making it of dimension 224*224*3 : Height, Width, Channel (RGB)
    img = image.load_img(image_path, target_size=inputShape)
    # Converting the image from PIL/Pillow format to Numpy Array
    img = image.img_to_array(img)
    # print(img.shape)
    # Since ResNet50 needs a 4D tensor as input for the convolutional neural network to work
    # therefore, we add one more dimension to convert it from (224,224,3) to (1,224,224,3)
    img = np.expand_dims(img, axis=0)
    # print(img.shape)
    # Preprocessing the images by subtracting the mean of each channel and thereby generating
    # the feature vector for our image
    features = imagenet_utils.preprocess_input(img)
    # Return the feature vector
    return features


# In[5]:

def prediction(features):
    # Given the features of our image, we predict the feature vector of the most close match
    # of our image using neural network i.e, the probabilities of being a part of some class
    predicted_features = model.predict(features)
    # Returns the List of Top 5 predicted classes out of 1000 for our given image based on probability
    return imagenet_utils.decode_predictions(predicted_features)


# In[5]:

# Generate Prediction Class for our image by passing the correct path of the image
img = sys.argv[1]
# P = prediction(preprocessing("./05-caption-images/COCO_val2014_000000002225.jpg"))
P = prediction(preprocessing(img))
# Output Top 5 predicted classes based on decreasing order of their probability
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


# In[ ]:

# For GUI using OpenCV
# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
if len(sys.argv) == 3 and sys.argv[2] == "GUI":
    orig = cv2.imread(img)
    (imagenetID, label, prob) = P[0][0]
    cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()
