from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import numpy as np
from keras.preprocessing import text

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import imdb

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 1000
config.maxlen = 1000
config.batch_size = 32
config.embedding_dims = 10
config.filters = 16
config.kernel_size = 3
config.hidden_dims = 256
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

print("Train:")
print("X_train shape: ", len(X_train))
print("y_train shape: ", len(y_train))

print("\nTest:")
print("X_test shape: ", len(X_test))
print("y_test shape: ", len(y_test))

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(config.hidden_dims, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])

# predict
y_pred = model.predict(X_test, verbose=0)
y_pred = y_pred[:, 0]
y_class_pred = np.rint(y_pred)

print("y_pred: ", y_pred)
print("y_class_pred: ", y_class_pred)

#accuracy
acc = accuracy_score(y_test, y_class_pred)
precision = precision_score(y_test, y_class_pred)
recall = recall_score(y_test, y_class_pred)
f1_Score = f1_score(y_test, y_class_pred)

print("accuracy: ", acc)
print("precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1_Score)
