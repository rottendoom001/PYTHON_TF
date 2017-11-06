import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


# //// DATOS DE LA RED NEURONAL ///////
n = 17  # Numero de característias de entrada
m = 1   # Numero de Salidas
h1 = 300   # Numero de neuronas en la capa oculta
h2 = 0
def read_test_data(filepath):
    # ////////// LEEMOS LOS DATOS DE PRUEBA /////////////
    df = pd.read_csv(
        filepath_or_buffer=filepath,
        header = None,
        sep=',')
    df.dropna(how="all", inplace=True) # drops the empty line at file-end
    ancho = len(df.columns)
    Xte = df.ix[:,0:ancho - 2].values
    Yte = df.ix[:, ancho - 1 ].values
    return Xte, Yte

def read_training_data(filepath):
    # //////////// LEEMOS TODOS LOS ELEMENTOS TRANSFORMADOS POR EL PCA /////////////
    df = pd.read_csv(
        filepath_or_buffer=filepath,
        header = None,
        sep=',')

    df.dropna(how="all", inplace=True) # drops the empty line at file-end
    ancho = len(df.columns)
    Xtr = df.ix[:,0:ancho - 2].values
    Ytr = df.ix[:, ancho - 1 ].values
    return Xtr, Ytr

def transform_tags_to_0_1(Y):
    k = 0
    while (k < len(Y)) :
        if Y[k].strip() == 'M':
            Y[k] = 0
        else :
            Y[k] = 1
        k+=1
    Y = Y.reshape((len(Y), m))
    return Y

def round_results(Y):
    k = 0
    while (k < len(Y)) :
        Y[k] = round(Y[k])
        k+=1
    return Y

def round(n):
    dec = n - int(n)
    rst = int(n) if dec < 0.5 else int(n)+1
    return rst

# /////////////// TRAINING ////////////////////
def training_model(Xtr):
    x_ = tf.placeholder(tf.float32, shape=[len(Xtr), n], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[len(Xtr), m], name="y-input")

    W1 = tf.Variable(tf.random_uniform([n,h1], -1, 1), name="W1")
    B1 = tf.Variable(tf.zeros([h1]), name="Bias1")
    output_1 = tf.sigmoid(tf.matmul(x_, W1) + B1)

    # 1 Capa oculta
    W2 = tf.Variable(tf.random_uniform([h1, m], -1, 1), name="W2")
    B2 = tf.Variable(tf.zeros([m]), name="Bias2")
    PRED = tf.sigmoid(tf.matmul(output_1, W2) + B2)
    return PRED, W1, B1, W2, B2, y_, x_

    # 2 Capas ocultas
    '''
    W2 = tf.Variable(tf.random_uniform([h1,h2], -1, 1), name="W2")
    B2 = tf.Variable(tf.zeros([h2]), name="Bias2")
    output_2 = tf.sigmoid(tf.matmul(output_1, W2) + B2)

    W3 = tf.Variable(tf.random_uniform([h2, m], -1, 1), name="W3")
    B3 = tf.Variable(tf.zeros([m]), name="Bias3")
    PRED = tf.sigmoid(tf.matmul(output_2, W3) + B3)
    return PRED, W1, B1, W2, B2, W3, B3, y_, x_
    '''




# /////////////// TEST MODEL ////////////////////
def test_model(W1, B1, W2, B2):
    xt_ = tf.placeholder(tf.float32, shape=[1 , n], name="x-input")
    output_1 = tf.sigmoid(tf.matmul(xt_, W1) + B1)

    # 1 Capa oculta
    PRED = tf.sigmoid(tf.matmul(output_1, W2) + B2)
    '''
    # 2 Capas ocultas
    output_2 = tf.sigmoid(tf.matmul(output_1, W2) + B2)
    PRED = tf.sigmoid(tf.matmul(output_2, W3) + B3)
    '''
    return PRED, xt_

# //////// LEEMOS LA MATRIZ DE PESOS DEL PCA  /////////////
def get_pca_w():
    df = pd.read_csv(
        filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/w_pca.csv',
        header = None,
        sep=',')

    df.dropna(how="all", inplace=True) # drops the empty line at file-end
    ancho = len(df.columns)
    W = df.ix[:,0:ancho - 1].values
    return W, ancho

# //////// KERAS MODEL  /////////////
def config_keras_model(input_shape, n_classes, epochs, lrate = 0.01):
    # define the architecture of the network
    model = Sequential()
    model.add(Dense(int(input_shape/4), input_dim=input_shape, init="uniform", activation="relu"))
    model.add(Dense(int(input_shape/8), init="uniform", activation="relu"))
    model.add(Dense(n_classes))
    ## 2 --> Number of final claseses
    model.add(Activation("softmax"))
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    ## lr = 0.01, learning rate
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model
