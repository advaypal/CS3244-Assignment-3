import numpy as np

TRAIN_SAMPLES = 'X_train.npy'
TRAIN_LABELS = 'y_train.npy'
TEST_SAMPLES = 'X_test.npy'
TEST_LABELS = 'y_test.npy'


def loadTrainData():
    return np.load(TRAIN_SAMPLES), np.load(TRAIN_LABELS)


def loadTestSamples():
    return np.load(TEST_SAMPLES)


def writeTestLabels(labels):
    np.save(TEST_LABELS, labels)


if __name__ == '__main__':
    trainSamples, trainLabels = loadTrainData()
    writeTestLabels(trainLabels)
    assert np.array_equal(np.load(TEST_LABELS), trainLabels)
