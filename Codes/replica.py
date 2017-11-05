import json
import embedding
import numpy as np
from keras import backend as K
from keras.utils.layer_utils import merge
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, GRU
from keras.models import Model
import matplotlib.pyplot as plt


def load(captions_filename, features_filename):
    features = np.load(features_filename)
    images = []
    texts = []
    with open(captions_filename) as fp:
        for line in fp:
            tokens = line.strip().split()
            images.append(tokens[0])
            texts.append(' '.join(tokens[1:]))
    return features, images, texts


features, images, texts = load(
    'annotations.10k.txt', 'resnet50-features.10k.npy')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
captions = pad_sequences(sequences, maxlen=16)
vocab = tokenizer.word_index
vocab['<eos>'] = 0  # add word with id 0

with open('vocab.json', 'w') as fp:  # save the vocab
    fp.write(json.dumps(vocab))

embedding_weights = embedding.load(
    vocab, 100, 'glove.twitter.27B.100d.filtered.txt')

image_input = Input(shape=(2048,))
caption_input = Input(shape=(16,))
noise_input = Input(shape=(16,))

caption_embedding = Embedding(
    len(vocab), 100, input_length=16, weights=[embedding_weights])
caption_rnn = GRU(256)
image_dense = Dense(256, activation='tanh')

image_pipeline = image_dense(image_input)
caption_pipeline = caption_rnn(caption_embedding(caption_input))
noise_pipeline = caption_rnn(caption_embedding(noise_input))

positive_pair = merge([image_pipeline, caption_pipeline], mode='dot')
negative_pair = merge([image_pipeline, noise_pipeline], mode='dot')
output = merge([positive_pair, negative_pair], mode='concat')


training_model = Model(
    input=[image_input, caption_input, noise_input], output=output)
image_model = Model(input=image_input, output=image_pipeline)
caption_model = Model(input=caption_input, output=caption_pipeline)


def custom_loss(y_true, y_pred):
    positive = y_pred[:, 0]
    negative = y_pred[:, 1]
    return K.sum(K.maximum(0., 1. - positive + negative))


def accuracy(y_true, y_pred):
    positive = y_pred[:, 0]
    negative = y_pred[:, 1]
    return K.mean(positive > negative)


training_model.compile(loss=custom_loss, optimizer='adam', metrics=[accuracy])


noise = np.copy(captions)
fake_labels = np.zeros((len(features), 1))
X_train = [features[:9000], captions[:9000], noise[:9000]]
Y_train = fake_labels[:9000]
X_valid = [features[-1000:], captions[-1000:], noise[-1000:]]
Y_valid = fake_labels[-1000:]

for epoch in range(10):
    np.random.shuffle(noise)  # don’t forget to shuffle mismatched captions
    history = training_model.fit(X_train, Y_train,
                                 validation_data=[
                                     X_valid, Y_valid], nb_epoch=1,
                                 batch_size=64)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accurary'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
