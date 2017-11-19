
# coding: utf-8

# In[2]:

from keras.models import load_model
import numpy as np
from Image_Feature_Extraction import extract_features


# In[10]:


# In[12]:

def generate_caption(image_filename, n=5):

    image_model = load_model('model.21epoch.image')
    caption_model = load_model('model.21epoch.caption')

    caption_representations = np.load('caption-representations-21epoch.npy')
    image_representations = np.load('image-representations-21epoch.npy')

    # generate image representation for new image
    image_representation = image_model.predict(
        extract_features(image_filename))
    # compute score of all captions in the dataset
    scores = np.dot(caption_representations, image_representation.T).flatten()
    # compute indices of n best captions
    indices = np.argpartition(scores, -n)[-n:]
    indices = indices[np.argsort(scores[indices])]
    output = []
    for i in [int(x) for x in reversed(indices)]:
        output.append(texts[i])
#         print(scores[i], texts[i])
    return output
