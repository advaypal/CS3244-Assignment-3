import sklearn.svm
import numpy as np
import data_utils
from sklearn.metrics import f1_score, precision_score, recall_score
import keras


def trainSVM(featureVectors, labels, kernel, cost=None,
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

def predictSVM(svmModel, featureVectors):
    return svmModel.predict(featureVectors)


def getMetrics(truthLabels, predLabels):
    return (f1_score(truthLabels, predLabels, average='samples'),
            precision_score(truthLabels, predLabels, average='samples'),
            recall_score(truthLabels, predLabels, average='samples'))


if __name__ == '__main__':
    x, y = data_utils.loadTrainData()

    # Shuffle to prevent overfitting validation
    p = np.random.permutation(len(x)); x = x[p]; y = y[p]

    xTrain, yTrain, xVal, yVal = data_utils.splitTrainVal(x, y, 4 * x.shape[0] // 5)

    model = trainSVM(xTrain, yTrain, kernel='linear')

    # Convert to one hot vectors since that's what sklearn metrics needs.
    yTrainCategorical = keras.utils.to_categorical(yTrain, num_classes=7)
    trainPredictions = predictSVM(model, xTrain)
    trainPredictions = keras.utils.to_categorical(trainPredictions, num_classes=7)

    print("\ttrain_f1: %f — train_precision: %f — train_recall %f"
          %(getMetrics(yTrainCategorical, trainPredictions)))

    # Convert to one hot vectors since that's what sklearn metrics needs.
    yValCategorical = keras.utils.to_categorical(yVal, num_classes=7)
    valPredictions = predictSVM(model, xVal)
    valPredictions = keras.utils.to_categorical(valPredictions, num_classes=7)

    print("\tval_f1: %f — val_precision: %f — val_recall %f"
          %(getMetrics(yValCategorical, valPredictions)))

    xTest = data_utils.loadTestSamples()
    testPredictions = predictSVM(model, xTest)
    data_utils.writeTestLabels(testPredictions)
