import keras
from   keras import layers as l
from   keras.models import Sequential
from   keras.layers.core import Dense, Activation, Flatten, Dropout
from   keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np  
import matplotlib.pyplot as plt
from   keras.datasets import fashion_mnist
from   tensorflow.keras.optimizers import Adam, Nadam
from   tensorflow.keras.regularizers import l2
import itertools
from   sklearn.metrics import confusion_matrix
from   sklearn.model_selection import train_test_split
from   sklearn.utils import shuffle
import scikitplot as skplt

#Hyperparameters
epochs     = 70
lr         = 0.0001
batch_size = 50

#Data pre-processing
(x_train , y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x = np.concatenate((x_train,x_test),axis=0)
y = np.concatenate((y_train,y_test),axis=0)

#Shuffle the dataset
x,y = shuffle(x, y, random_state=2)
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=2)

#Defining the NN Architecture
model = Sequential([
Convolution2D(filters = 32, kernel_size = 3, strides = 1, activation = 'relu', input_shape = (28,28,1), kernel_regularizer=l2(0.0005)),
Convolution2D(filters = 32, kernel_size = 3, strides = 1, use_bias=False),
l.BatchNormalization(),
Activation('relu'),
MaxPooling2D(pool_size = 2, strides = 2),
Dropout(0.25),
Convolution2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=l2(0.0005)),
Convolution2D(filters = 64, kernel_size = 3, strides = 1, use_bias=False),
l.BatchNormalization(),
Activation('relu'),
MaxPooling2D(pool_size = 2, strides = 2),
Dropout(0.25),
Flatten(),
# ----------------------------------------------------------------------#
Dense(units = 1024, use_bias=False),
l.BatchNormalization(),
Activation('relu'),
Dropout(0.1),
Dense(units = 512, use_bias=False),
l.BatchNormalization(),
Activation('relu'),
Dropout(0.25),
Dense(units = 256, use_bias=False),
l.BatchNormalization(),
Activation('relu'),
Dropout(0.5),
Dense(units = 10, activation = 'softmax')
])

filepath = 'weights_checkpoint'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath = filepath,
                                                             save_weights_only=True,
                                                             monitor='accuracy',
                                                             mode='max',
                                                             save_best_only=False)

def training_step(epochs,lr):
    model.compile(optimizer=Nadam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    x = model.fit(x_train,
                  y_train,
                  validation_split=0.1,
                  epochs=epochs,
                  shuffle=True,
                  batch_size=batch_size,
                  verbose=2)
    return x
        
history =  training_step(epochs, lr)
model.save_weights('weights')

val_loss, val_acc = model.evaluate(x_test, y_test)
print('Loss: ' + str(round(val_loss,5)) + '  |  ' +  'Accuracy: ' + str(round(val_acc,5)))

pred = model.predict_classes(x=x_test)
p    = model.predict(x=x_test)
cm   = confusion_matrix(y_true=y_test,y_pred=pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix,without normalization')
        
    print(cm)
    
    tresh= cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i,j] > tresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cm.png', dpi = 300)
    

cm_plot_labels = ['Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9','Class 10']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title='Confusion Matrix')

#Plotting the graphs
def plot_accuracy():
    plt.figure(figsize=(20,10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('MODEL ACCURACY (eta='+ str(lr) + ')')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.savefig('accuracy.jpg',dpi=300)
    plt.show()
def plot_cost():
    plt.figure(figsize=(20,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MODEL LOSS (eta='+ str(lr) + ')')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.savefig('cost.png', dpi = 300)
    plt.show()

plot_accuracy()
plot_cost()

# PLot precision recall and ROC
skplt.metrics.plot_precision_recall(y_test,p)
skplt.metrics.plot_roc(y_test,p)

