'''
Keras official mnist cnn example, adapted to our dataset
'''
import data_io
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#         
# datagen.fit(x_train)
# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                     steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
#                     verbose=1)

NUM_CLASSES, IMAGE_ROWS, IMAGE_COLS = 7, 50, 37

def augmentData(x_train, y_train, x_test):
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    return (x_train, y_train, x_test)


def cnnTrain(x, y, batch_size=32, epochs=15):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2)

def cnnPredict(x, model):
    return model.predict(x)

if __name__ == '__main__':
    (inital_x_train, initial_y_train) = data_io.loadTrainData()
    initial_x_test =  data_io.loadTestSamples()
    (x_train, y_train, x_test) = augmentData(initial_x_train, initial_y_train, initial_x_test)
    model = cnnTrain(x_train, y_train)
    # predictions = cnnPredict(x_test, model)
    # data_io.writeTestLabels(predictions)
