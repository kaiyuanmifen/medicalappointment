#mortality prediction
from keras.models import Sequential
from keras.models import load_model
from keras import models
from keras import layers
from keras.models import Model
from keras import optimizers
from keras.layers import Dense, Activation
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras import regularizers
import keras.backend as K
import tensorflow as tf



# #trak AUCROC each epoch

import tensorflow as tf
from keras import backend as K

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def PredictiveNeuralNetwork_binary(Inputshape,OutputShape=1):
    print("version1")


    neuralnetwork=Sequential([
        Dense(128, activation='relu', input_dim=Inputshape),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(OutputShape, activation='sigmoid')
    ])


    neuralnetwork.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=[auc])

    print("Neural network with " + str(neuralnetwork.count_params()) + " Parameters and " + str(
        len(neuralnetwork.layers)) + " Layers")
    print("Neural network model generated")

    return neuralnetwork



def PredictiveNeuralNetwork(Inputshape,OutputShape=3):
    print("version1")


    neuralnetwork=Sequential([
        Dense(128, activation='relu', input_dim=Inputshape),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(OutputShape, activation='softmax')
    ])


    neuralnetwork.compile(optimizer='Nadam', loss='categorical_crossentropy', shuffle=True, metrics=[auc])

    print("Neural network with " + str(neuralnetwork.count_params()) + " Parameters and " + str(
        len(neuralnetwork.layers)) + " Layers")
    print("Neural network model generated")

    return neuralnetwork


def LogisticRegression(Inputshape):

    neuralnetwork=Sequential([
        Dense(1,input_dim=Inputshape, activation='sigmoid')
    ])

    neuralnetwork.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=[auc])

    print("logistic regression with " + str(neuralnetwork.count_params()) + " Parameters and " + str(
        len(neuralnetwork.layers)) + " Layers")
    print("logistic regression model generated")

    return neuralnetwork



def RegressionModel(Inputshape):

    neuralnetwork=Sequential([
        Dense(1,input_dim=Inputshape, activation='sigmoid')
    ])

    neuralnetwork.compile(optimizer='Nadam', loss='mse', shuffle=True)

    print("lregression with " + str(neuralnetwork.count_params()) + " Parameters and " + str(
        len(neuralnetwork.layers)) + " Layers")
    print(" regression model generated")

    return neuralnetwork