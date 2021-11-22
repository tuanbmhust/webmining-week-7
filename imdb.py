import numpy as np
import os
import random

sep = os.path.sep

nSample = min(17012-6000, 12500)

def load_imdb():
    X_train = []
    y_train = []

    path = os.path.join('aclImdb', 'train', 'pos', '')
    # X_train.extend([open(path + f, encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    # y_train.extend([1 for _ in range(12500)])
    X_train.extend(random.sample([open(path + f, encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')], nSample))
    y_train.extend([1 for _ in range(nSample)])

    path = os.path.join('aclImdb', 'train', 'neg', '')
    # X_train.extend([open(path + f, encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    # y_train.extend([0 for _ in range(12500)])
    X_train.extend(random.sample([open(path + f, encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')], nSample))
    y_train.extend([0 for _ in range(nSample)])

    X_test = []
    y_test = []

    path = os.path.join('aclImdb', 'test', 'pos', '')
    X_test.extend([open(path + f, encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([1 for _ in range(12500)])

    path = os.path.join('aclImdb', 'test', 'neg', '')
    X_test.extend([open(path + f, encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([0 for _ in range(12500)])

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    return (X_train, y_train), (X_test, y_test)
