import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.regularizers import l2
import data_io

import matplotlib.pyplot as plt
import numpy as np

# Training and Validation Split
B = 128  # Batch Size
N = B * 6  # Training Samples
E = 200  # Epochs


def buildFCNN():
    # Architecture
    model = Sequential()
    model.add(BatchNormalization(input_shape=(1850,)))
    for i in range(5):
        model.add(Dense(1024, activation='relu', W_regularizer=l2(0.01)))
        model.add(Dropout(0.03))
    model.add(Dense(7, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    # plot_model(model, to_file='model.png')  # Generate Image of Architecture

    return model


def trainFCNN(model, datagen, xTrain, yTrain, xVal, yVal):
    # Training
    def gen():
        for x_batch, y_batch in datagen.flow(xTrain, yTrain, batch_size=B):
            yield (x_batch.reshape(B, 1850), y_batch)

    model.fit_generator(gen(), steps_per_epoch=N//B, epochs=E, verbose=2,
                        validation_data=(xVal, yVal))



def outputModel(model, xTest):
    # If 'Enter', Create Test Predictions File
    input('continue...')

    model.save('model.h5')  # Save Model Architecture and Weights

    f = open('predictions.txt', 'w')
    print('ImageId,PredictedClass')
    f.write('ImageId,PredictedClass\n')
    for i, p in enumerate(np.argmax(model.predict(xTest), axis=1)):
        print('%d,%d' % (i, p))
        f.write('%d,%d\n' % (i, p))
    f.close()

if __name__ == '__main__':
    x, y = data_utils.loadTrainData()
    x = x.reshape(-1, 50, 37, 1)
    y = keras.utils.to_categorical(y, num_classes=7)
    xTrain, yTrain, xVal, yVal = data_utils.splitTrainVal(x, y, N)

    datagen = data_utils.augmentData(xTrain)
    data_utils.standardizeData(xVal)

    model = buildFCNN()
    trainFCNN(model, datagen, xTrain, yTrain, xVal, yVal)

    xTest = data_utils.loadTestSamples()
    xTest = xTest.reshape(-1, 50, 37, 1)
    data_utils.standardizeData(xTest)

    outputModel(model, xTest)
