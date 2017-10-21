# Unable to figure out how to import data_io from this file (without changing
# sys path), so simply rewriting that stuff here. If anyone has suggestions,
# I would be glad to hear them.

import sklearn.svm
import numpy as np


def svmTrain(featureVectors, labels, kernel, cost=None,
             gamma=None, degree=None, coef0=None):
    # The following assignments are needed to stop the model from crashing.
    # Each variable is assigned to the default value used by the library
    if cost is None:
        cost = 1.0

    if degree is None:
        degree = 3

    if coef0 is None:
        coef0 = 0.0

    if gamma is None:
        gamma = 'auto'

    svmModel = sklearn.svm.SVC(
        C=cost,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0
    )

    svmModel.fit(featureVectors, labels)

    return svmModel

def svmPredict(featureVectors, svmModel):
    return svmModel.predict(featureVectors)


if __name__ == '__main__':
    trainSamples, trainLabels = np.load('data/X_train.npy'), np.load('data/y_train.npy')
    testSamples = np.load('data/X_test.npy')

    model = svmTrain(trainSamples, trainLabels, kernel='linear')
    predictions = svmPredict(testSamples, model)
    np.savetxt(
        'data/y_test.txt',
        np.dstack((np.arange(predictions.size), predictions))[0],
        delimiter=',',
        header='ImageId,PredictedClass',
        fmt='%d',
        comments=''
    )
