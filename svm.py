import sklearn.svm
import numpy as np
import data_io


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
    trainSamples, trainLabels = data_io.loadTrainData()
    testSamples = data_io.loadTestSamples()

    model = svmTrain(trainSamples, trainLabels, kernel='linear')
    predictions = svmPredict(testSamples, model)
    data_io.writeTestLabels(predictions)
