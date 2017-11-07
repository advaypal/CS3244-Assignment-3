import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.regularizers import l2
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import numpy as np
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import data_utils

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# Training and Validation Split ------------------------------------------------
B = 128  # Batch Size
N = B * 6  # Training Samples
E = 700  # Epochs


def buildModel():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(50,37,1)))
    model.add(Conv2D(
        64, kernel_size=(7, 7), padding='valid', activation='relu', W_regularizer=l2(0.01)))
    model.add(Conv2D(
        64, (7, 7), activation='relu', padding='valid', W_regularizer=l2(0.001)))
    model.add(Dropout(0.01))
    #model.add(Conv2D(
    #    64, kernel_size=(5, 5), padding='same', activation='relu', W_regularizer=l2(0.008)))
    #model.add(Dropout(0.01))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', W_regularizer=l2(0.005)))
    model.add(Dropout(0.01))
    model.add(Dense(64, activation='relu', W_regularizer=l2(0.005)))
    model.add(Dropout(0.005))
    model.add(Dense(7, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    # plot_model(model, to_file='model.png')  # Generate Image of Architecture

    return model


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (
            np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='samples')
        _val_recall = recall_score(val_targ, val_predict, average='samples')
        _val_precision = precision_score(val_targ, val_predict, average='samples')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("\tval_f1: %f — val_precision: %f — val_recall %f"
              %(_val_f1, _val_precision, _val_recall))


def trainCNN(model, datagen, xTrain, yTrain, xVal, yVal):
    def gen():
        for x_batch, y_batch in datagen.flow(xTrain, yTrain, batch_size=B):
            yield x_batch, y_batch

    metrics = Metrics()
    metrics.validation_data = (xVal, yVal)

    # ------------------- End of Code from the Internet

    #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=B,
    #                          write_graph=False, write_grads=True, write_images=True)
    model.fit_generator(gen(), steps_per_epoch=N//B, epochs=E, verbose=2,
                        validation_data=(xVal, yVal),
                        callbacks=[metrics])


def outputModelAndPredictions(model, xTest):
    # If 'Enter', Create Test Predictions File
    input('continue...')

    model.save('model.h5')  # Save Model Architecture and Weights

    data_utils.writeTestLabels(np.argmax(model.predict(xTest), axis=1))

if __name__ == '__main__':
    ktf.set_session(get_session())

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
