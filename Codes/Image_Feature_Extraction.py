
# coding: utf-8

# In[1]:

#Imports
from resnet50 import ResNet50
from imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np


# In[2]:

resnet = ResNet50(weights="imagenet", include_top=False)


# In[3]:

def extract_features(img_path):
    # load the input image using the Keras helper utility while ensuring
    # that the image is resized to 224x224 pxiels, the required input
    # dimensions for the network -- then convert the PIL image to a
    # NumPy array
    img = image.load_img(img_path, target_size=(224, 224))
    # Since the img is of type PIL/Pillow, so we need to convert it to numpy array
    x = image.img_to_array(img)
    # our image is now represented by a NumPy array of shape (3, 224, 224),
    # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
    # pass it through the network -- we'll also preprocess the image by
    # subtracting the mean RGB pixel intensity from the ImageNet dataset
    x = np.expand_dims(x, axis=0)
    # 
    x = preprocess_input(x)
    features = resnet.predict(x)
    return np.expand_dims(features.flatten(), axis=0)


# In[4]:

print(extract_features("./05-caption-images/COCO_val2014_000000000923.jpg").shape)
extract_features("./05-caption-images/COCO_val2014_000000000923.jpg")

