import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


NUM_DATA_POINTS = 20


def getData(trainFunction, predictFunction, samples, labels):
    trainSamples, testSamples, trainLabels, testLabels = train_test_split(
        samples, labels, test_size=0.2
    )

    indices = list(range(trainLabels.shape[0]))
    random.shuffle(indices)

    data = []

    for numSamples in np.linspace(10, trainLabels.shape[0], num=NUM_DATA_POINTS):
        numSamples = int(numSamples)
        trainSamplesToUse = trainSamples[indices[ : numSamples]]
        trainLabelsToUse = trainLabels[indices[ : numSamples]]

        model = trainFunction(trainSamplesToUse,
                              trainLabelsToUse)

        trainPredictions = predictFunction(model, trainSamplesToUse)
        eIn = (np.count_nonzero(trainPredictions != trainLabelsToUse) /
               trainLabelsToUse.shape[0])

        testPredictions = predictFunction(model, testSamples)
        eOut = (np.count_nonzero(testPredictions != testLabels) /
                testLabels.shape[0])

        data.append([numSamples, eIn, eOut])

    return zip(*data)


def plotLearningCurve(trainFunction, predictFunction, samples, labels):
    numSamplesUsed, eIns, eOuts = getData(trainFunction, predictFunction,
                                          samples, labels)
    plt.plot(numSamplesUsed, eIns, 'bx-', label='EIn')
    plt.plot(numSamplesUsed, eOuts, 'rx-', label='EOut')
    plt.legend(loc='upper right')
    plt.title('Learning Curve')
    plt.xlabel('Number of training samples')
    plt.ylabel('Error rate')
    plt.show()
