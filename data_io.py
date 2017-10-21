import os.path
import numpy as np

DATA_FOLDER = 'data'
TRAIN_SAMPLES = 'X_train.npy'
TRAIN_LABELS = 'y_train.npy'
TEST_SAMPLES = 'X_test.npy'
TEST_LABELS = 'y_test.npy'


def loadTrainData():
    return (np.load(os.path.join(DATA_FOLDER, TRAIN_SAMPLES)),
            np.load(os.path.join(DATA_FOLDER, TRAIN_LABELS)))


def loadTestSamples():
    return np.load(os.path.join(DATA_FOLDER, TEST_SAMPLES))


def writeTestLabels(labels):
    np.save(os.path.join(DATA_FOLDER, TEST_LABELS), labels)


if __name__ == '__main__':
    trainSamples, trainLabels = loadTrainData()
    print(trainSamples.shape, trainLabels.shape)

    writeTestLabels(trainLabels)
    assert np.array_equal(np.load(os.path.join(DATA_FOLDER, TEST_LABELS)),
                          trainLabels)
