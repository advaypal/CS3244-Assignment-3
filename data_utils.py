import os.path
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

DATA_FOLDER = 'data'
TRAIN_SAMPLES = 'X_train.npy'
TRAIN_LABELS = 'y_train.npy'
TEST_SAMPLES = 'X_test.npy'
TEST_LABELS = 'predictions.txt'

DATAGEN = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True)


def loadTrainData():
    return (np.load(os.path.join(DATA_FOLDER, TRAIN_SAMPLES)),
            np.load(os.path.join(DATA_FOLDER, TRAIN_LABELS)))


def loadTestSamples():
    return np.load(os.path.join(DATA_FOLDER, TEST_SAMPLES))


def writeTestLabels(labels):
    np.savetxt(
        os.path.join(DATA_FOLDER, TEST_LABELS),
        np.dstack((np.arange(labels.size), labels))[0],
        delimiter=',',
        header='ImageId,PredictedClass',
        fmt='%d',
        comments=''
    )


def splitTrainVal(samples, labels, trainSplit):
    return samples[:trainSplit], labels[:trainSplit], samples[trainSplit:], labels[trainSplit:]


def augmentData(samples):
    DATAGEN.fit(samples)
    return DATAGEN


def standardizeData(samples):
    DATAGEN.standardize(samples)
