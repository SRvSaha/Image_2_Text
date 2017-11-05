#
#   @author      : SRvSaha
#   Filename     : text_preprocessing.py
#   Timestamp    : 12:12 03-November-2017 (Saturday)
#   Email        : contact [dot] srvsaha [at] gmail.com

import sys
import numpy as np
import json
# Tokenizer is a class which is a high level
# API in keras for dealing with text preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_features_images_captions(captions_file, features_file):
    # For loading the image features extracted using ResNet50
    features = np.load(features_file)
    # To store the id(name) of the images
    images = []
    # To store the corresponding caption of the images
    texts = []
    # The Input Captions file consists of Image Name and the
    # corresponding caption separated by space
    with open(captions_file) as f:
        # For each line in the captions file
        for line in f:
            # Removing the trailing and leading whitespaces
            # and splitting using space so that there are two
            # token : Image Name and the corresponding caption
            tokens = line.strip().split()
            # Image names are inserted into the list of images
            images.append(tokens[0])
            # Since all the words of the caption are also separated
            # by space so they are joined again to recompose the string
            # and then caption in inserted into the list of texts
            texts.append(" ".join(tokens[1:]))
    # Returns the Image Features, Image Name and Corresponding Caption
    return features, images, texts


def preprocess_text(texts):
    # Instantiating the Tokenizer Class
    tokenizer = Tokenizer()
    # Fitting the tokenizer object on List of the texts/captions
    tokenizer.fit_on_texts(texts)
    # Converts each word to integer sequence i.e, basically it assigns
    # an integer id to each word so, duplicates of any kind is removed
    sequences = tokenizer.texts_to_sequences(texts)
    # To check what is the length of the sequence with the max vector
    # so that we can have an ideas what should be the maximum length of the
    # padding that we are using
    max_sequence_length = max([len(seq) for seq in sequences])
    # Padding is done since all the caption are not of same length
    # All the captions were observed to be small so, size of 16 is chosen
    # for the overall unified length of all captions.
    # If the length of the caption is less than 16, it is padded with zeros
    # in the front (by default)
    captions_after_padding = pad_sequences(sequences, maxlen=16)
    # word.index returns a dictionary where the key is the word and the value
    # is the index that is assigned/ by which the word sequence is changed into
    # integer sequence
    vocabulary = tokenizer.word_index
    # Adding the entry of end of sentence into the dictionary as a values of 0
    # so as to keep track of the end of the sentence.
    vocabulary['<eos>'] = 0
    # Returns the numpy array of shape (10000,16) where each input is converted
    # to vector of 16 dimension
    return captions_after_padding


def generate_and_save_vocabulary(texts):
    # Instantiating the Tokenizer Class
    tokenizer = Tokenizer()
    # Fitting the tokenizer object on List of the texts/captions
    tokenizer.fit_on_texts(texts)
    # Converts each word to integer sequence i.e, basically it assigns
    # an integer id to each word so, duplicates of any kind is removed
    sequences = tokenizer.texts_to_sequences(texts)
    # word.index returns a dictionary where the key is the word and the value
    # is the index that is assigned/ by which the word sequence is changed into
    # integer sequence
    vocabulary = tokenizer.word_index
    # Adding the entry of end of sentence into the dictionary as a values of 0
    # so as to keep track of the end of the sentence.
    vocabulary['<eos>'] = 0
    with open('vocabulary.json', 'w') as f:
        f.write(json.dumps(vocabulary))
    print("Successfully dumped as JSON :)")


if __name__ == "__main__":
    features, images, texts = load_features_images_captions(
        sys.argv[1], sys.argv[2])
    print(preprocess_text(texts))
    generate_and_save_vocabulary(texts)
