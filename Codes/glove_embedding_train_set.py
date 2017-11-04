#
#   @author      : SRvSaha
#   Filename     : glove_embedding_train_set.py
#   Timestamp    : 12:05 03-November-2017 (Saturday)

import embedding
import sys
import json

# Generating the GloVe embeddings on the train data where each word in our vocab
# gets the GloVe Embedding of 100 dimension which is trained on 27B words in
# Twitter Data.
# Basically we need to map 16 dimension input to 100 dimension vector representation
# Therefore from out vocabulary we build matrix of shape (len(vocab) + 1, 100)
# and thus we will use the Embedding module of Keras to project the 16 dimension
# input to 100 dimension word embeddings.
# arg 1 : vocabulary
# arg 2 : Dimension of output embeddings
# arg 3 : Filename of the Original Glove Embedding

with open(sys.argv[1]) as f:
    # This is used to load the json file and convert it to Python Dictionary
    vocabulary = json.load(f)

# Converting the word in our vocabulary to 100 dimension as in GloVe Embedding
# Now basically all our words in the vocabulary are converted to 100 dimension
# embedding which will be passed to GRU (RNN is to be used).
embedding_weights = embedding.load(vocabulary, sys.argv[2], sys.argv[3])
# The shape will be (length(vocabulary), dimension) which means that each word
# is now represented by GloVe of 100 dimension
print(embedding_weights)
