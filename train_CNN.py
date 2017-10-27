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

# Training and Validation Split ------------------------------------------------
B = 128  # Batch Size
N = B * 6  # Training Samples
E = 700  # Epochs

x = np.load('data/X_train.npy', 'c')
y = keras.utils.to_categorical(np.load('data/y_train.npy', 'r'), num_classes=7)
# Shuffle to prevent overfitting validation
p = np.random.permutation(len(x)); x = x[p]; y = y[p]  

x_train = x[:N]
x_train_rs = x_train.reshape(N, 50, 37, 1)
x_val = x[N:]
x_val_rs = x_val.reshape(x_val.shape[0], 50, 37, 1)

y_train = y[:N]
y_val = y[N:]

# Data Augmentation and Standarization  ----------------------------------------
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
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

# Architecture  ----------------------------------------------------------------
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
#plot_model(model, to_file='model.png')  # Generate Image of Architecture

# Training  --------------------------------------------------------------------
def gen():
    for x_batch, y_batch in datagen.flow(x_train_rs, y_train, batch_size=B):
        yield x_batch, y_batch
datagen.standardize(x_val_rs)

# Remove when Submiting? (val_f1) --------------------
from keras.callbacks import Callback 
from sklearn.metrics import f1_score, precision_score, recall_score

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
        #if _val_f1 > NUM:
        #    print('Early Stop!')
        #    self.model.stop_training = True
 
metrics = Metrics()
metrics.validation_data = (x_val_rs,y_val)

# ------------------- End of Code from the Internet

#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=B,
#                          write_graph=False, write_grads=True, write_images=True)
model.fit_generator(gen(), steps_per_epoch=N//B, epochs=E, verbose=2,
                    validation_data=(x_val_rs,y_val),
                    callbacks=[metrics])

# If 'Enter', Create Test Predictions File -------------------------------------
input('continue...')
model.save('model.h5')  # Save Model Architecture and Weights
x_test = np.load('data/X_test.npy', 'c')
x_test_rs = x_test.reshape(x_test.shape[0], 50, 37, 1)
datagen.standardize(x_test_rs)
f = open('predictions.txt', 'w')
print('ImageId,PredictedClass')
f.write('ImageId,PredictedClass\n')
for i, p in enumerate(np.argmax(model.predict(x_test_rs), axis=1)):
    print('%d,%d' % (i, p))
    f.write('%d,%d\n' % (i, p))
f.close()
