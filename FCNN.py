import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.regularizers import l2

import matplotlib.pyplot as plt
import numpy as np

# Training and Validation Split
B = 128  # Batch Size
N = B * 6  # Training Samples
E = 200  # Epochs

x = np.load('data/X_train.npy', 'c')
x_train = x[:N]
x_train_rs = x_train.reshape(N, 50, 37, 1)
x_val = x[N:]
x_val_rs = x_val.reshape(x_val.shape[0], 50, 37, 1)

y = keras.utils.to_categorical(np.load('data/y_train.npy', 'r'), num_classes=7)
y_train = y[:N]
y_val = y[N:]

# Data Augmentation and Standarization
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True)
datagen.fit(x_train_rs)

# Uncomment to see Raw vs Standarized Data
#for image in x_train_rs:
#    fig = plt.figure()
#    fig.add_subplot(1, 2, 1).set_title('Raw')
#    plt.imshow(image.reshape(50, 37), cmap=plt.cm.Greys, interpolation='none')
#    fig.add_subplot(1, 2, 2).set_title('Standarized')
#    # NOTE: standarize changes the original matrix,
#    #       do not train with double standarization.
#    plt.imshow(datagen.standardize(image).reshape(50, 37),
#               cmap=plt.cm.Greys, interpolation='none')
#    plt.show()

# Uncomment to see Augmented Data
#for batch in datagen.flow(x_train_rs, y_train, batch_size=128):
#    for image in batch[0]:
#        plt.imshow(image.reshape(50, 37), cmap=plt.cm.Greys, interpolation='none')
#        plt.show()

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
plot_model(model, to_file='model.png')  # Generate Image of Architecture

# Training
def gen():
    for x_batch, y_batch in datagen.flow(x_train_rs, y_train, batch_size=B):
        yield (x_batch.reshape(B, 1850), y_batch)
datagen.standardize(x_val_rs)
model.fit_generator(gen(), steps_per_epoch=N//B, epochs=E, verbose=2,
                    validation_data=(x_val,y_val))
model.save('model.h5')  # Save Model Architecture and Weights

# If 'Enter', Create Test Predictions File
input('continue...')
x_test = np.load('data/X_test.npy', 'c')
x_test_rs = x_test.reshape(x_test.shape[0], 50, 37, 1)
datagen.standardize(x_test_rs)
f = open('predictions.txt', 'w')
print('ImageId,PredictedClass')
f.write('ImageId,PredictedClass\n')
for i, p in enumerate(np.argmax(model.predict(x_test), axis=1)):
    print('%d,%d' % (i, p))
    f.write('%d,%d\n' % (i, p))
f.close()
