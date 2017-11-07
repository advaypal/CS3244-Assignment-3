import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.regularizers import l2
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import data_utils


B = 128  # Batch Size
N = B * 6  # Number of Training Samples
E = 800  # Number of Epochs


def buildModel():
    # Architecture
    model = Sequential()
    model.add(BatchNormalization(input_shape=(50,37,1)))
    model.add(Conv2D(
        64, kernel_size=(7, 7), padding='valid', activation='relu', W_regularizer=l2(0.01)))
    model.add(Conv2D(
        64, (7, 7), activation='relu', padding='valid', W_regularizer=l2(0.001)))
    model.add(Dropout(0.06))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', W_regularizer=l2(0.005)))
    model.add(Dropout(0.05))
    model.add(Dense(64, activation='relu', W_regularizer=l2(0.005)))
    model.add(Dropout(0.07))
    model.add(Dense(7, activation='softmax'))

    # Create optimizer
    sgd = SGD(lr=0.015, decay=1e-6, momentum=0.7, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # plot_model(model, to_file='model.png')  # Generate Image of Architecture

    return model


# Metrics class to get validation metrics during training.
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):       
        val_predict = (
            np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='samples')
        
        print('val_f1: %f -' % _val_f1, end=' ')


def trainCNN(model, datagen, xTrain, yTrain, xVal, yVal):
    metrics = Metrics()
    metrics.validation_data = (xVal, yVal)

    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=B,
    #                           write_graph=False, write_grads=True, write_images=True)

    # Train model
    model.fit_generator(
        datagen.flow(xTrain, yTrain, batch_size=B), steps_per_epoch=N//B,
        epochs=E, verbose=2, validation_data=(xVal, yVal), callbacks=[metrics])


def outputModelAndPredictions(model, xTest):
    # If 'Enter', Create Test Predictions File
    input('Press Enter to continue...')

    model.save('model.h5')  # Save Model Architecture and Weights

    data_utils.writeTestLabels(np.argmax(model.predict(xTest), axis=1))


if __name__ == '__main__':
    x, y = data_utils.loadTrainData()

    # Shuffle to prevent overfitting validation
    p = np.random.permutation(len(x)); x = x[p]; y = y[p]

    x = x.reshape(-1, 50, 37, 1)
    y = keras.utils.to_categorical(y, num_classes=7)
    xTrain, yTrain, xVal, yVal = data_utils.splitTrainVal(x, y, N)

    datagen = data_utils.augmentData(xTrain)
    data_utils.standardizeData(xVal)

    model = buildModel()
    trainCNN(model, datagen, xTrain, yTrain, xVal, yVal)

    xTest = data_utils.loadTestSamples()
    xTest = xTest.reshape(-1, 50, 37, 1)
    data_utils.standardizeData(xTest)

    outputModelAndPredictions(model, xTest)
