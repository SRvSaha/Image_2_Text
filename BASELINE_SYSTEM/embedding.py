import sys
import numpy as np


def load(vocab, dimension, filename):
    print('loading embeddings from "%s"' % filename, file=sys.stderr)
    dimension = int(dimension)
    # Creating a numpy 2D matrix with dimension len(vocab)*dimension.
    # Each row will consist of 100 dimension vectors
    embedding = np.zeros((len(vocab), dimension), dtype=np.float32)
    # seen is the set that keeps track of the works that are in vocabulary
    # and seen in the GloVe Embedding
    seen = set()
    with open(filename) as fp:
        for line in fp:
            tokens = line.strip().split(' ')
            # if the len(token) is equal to dimension + 1 since the first token
            # will be the word
            if len(tokens) == dimension + 1:
                # first token is the word
                word = tokens[0]
                # If the word is found in vocabulary keys
                if word in vocab:
                    # key is the UNIQUE INDEX of the word and value is the list
                    # of the vector of dimension as passed.
                    embedding[vocab[word]] = [float(x) for x in tokens[1:]]
                    # Adding the word to seen set as it is encountered
                    seen.add(word)
                    # When all the words will be processed then length of seen
                    # and length of vocabulary will be same. We break out of the
                    # loop and return the embedding
                    if len(seen) == len(vocab):
                        break
    return embedding


if __name__ == '__main__':
    # can be used to filter an embedding file
    # This is used when the script is used independently
    if len(sys.argv) != 3:
        print('usage: cat wordlist | %s <dimension> <embedding_filename>' %
              sys.argv[0])
        sys.exit(1)

    vocab = {word.strip(): i for i, word in enumerate(sys.stdin.readlines())}
    dimension = int(sys.argv[1])
    filename = sys.argv[2]
    embedding = load(vocab, dimension, filename)

    for word, i in vocab.items():
        print(word, ' '.join([str(x) for x in embedding[i]]))
