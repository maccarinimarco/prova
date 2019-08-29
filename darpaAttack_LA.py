import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 

n_classi=3
dsTrain = pd.read_csv('dataset/DARPA_training_set_3classi_small.csv')
dsTest= pd.read_csv('dataset/DARPA_test_set_3classi_small.csv')
train_images = dsTrain.drop(columns=['ClassValues','protocol_type','service','flag'])
train_labels = dsTrain[['ClassValues']]
test_images = dsTest.drop(columns=['ClassValues','protocol_type','service','flag'])
test_labels = dsTest[['ClassValues']]
 


from keras.utils import to_categorical
#converto i labels
#to_categorical trasforma per mezzo di one hot encoding
train_labels = to_categorical(train_labels-1,num_classes=n_classi)
test_labels = to_categorical(test_labels-1,num_classes=n_classi)

import tensorflow as tf
from keras import layers
from keras import models
from tensorflow import keras
network = keras.Sequential([
keras.layers.Dense(512, activation='relu', input_shape=(38 ,)),
keras.layers.Dense(n_classi, activation='softmax')])
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#eseguo il 'resize'




history =  network.fit(train_images, train_labels,  epochs=5, batch_size=100, verbose=1)
 
test_loss, test_acc = network.evaluate(test_images, test_labels)
 
print('test_acc:', test_acc)

if test_acc > 0.8:
    network.save('rete{}.m'.format(test_acc))

'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuratezzaModelloMNIST.pdf')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lossFunctionMNIST.pdf')
plt.show()

'''