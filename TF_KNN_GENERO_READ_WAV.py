import numpy as np
import wave
import sys
from scipy import fft, arange

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def printNvalues(n, arr) :
    i = 0
    for v in arr :
        print ("[{}] = {}".format(i,v))
        if n == i :
            break
        i+=1

def format2CSV(n, arr, g) :
    s = ''
    i = 0
    for v in arr :
        s = s + str(v[1]) + ", "
        if n == i :
            break
        i+=1
    s = s + g + '\n'
    return s

def getMaximumFreq(n, arr) :
    max_freq = []
    i = 0
    for v in arr :
        max_freq.append(v[1])
        if n == i :
            break
        i+=1
    return max_freq


def calculateSpectrumWithHPS(y,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    j = int (n/2)
    frq = frq[range(j)] # one side frequency range
    Y = fft(y)/n # fft computing and normalization
    Y[0] = 0
    #printNvalues(10,abs(Y))
    Y = Y[range(j)]
    Y = abs(Y)
    Y = hps(Y)
    return Y, frq

def save2CSV(name, value):
    archivo = open(name, 'a')
    archivo.write(value)
    archivo.close()

def hps(arr):
    r = arr
    d2 = []
    d3 = []
    i = 0
    # Diesmar en 2
    for v in arr :
        if  i % 2 == 0 :
            d2.append(v)
        i+=1
    #Diesmar en 3
    i = 0
    for v in arr :
        if  i % 3 == 0 :
            d3.append(v)
        i+=1
    d2 = np.array(d2)
    d3 = np.array(d3)

    #print("r : ", r.size)
    #print("d2 : ", d2.size)
    #print("d3 : ", d3.size)

    #Multiplicar por d2
    i = 0
    for v in d2 :
        r[i] = r[i] * v
        i+=1
    #Multiplicar por d3
    i = 0
    for v in d3 :
        r[i] = r[i] * v
        i+=1
    return r

def clearSignal(signal):
    maxValue = np.amax(signal)
    limit = maxValue/4
    n = len (signal) - 1
    i = 0
    f = 0
    # Buscamos desde donde comenzar
    while i <= n:
        if signal[i] >= limit:
            break
        i+=1
    # Buscamos donde terminar
    while f <= n:
        if signal[n - f] >= limit:
            break
        f+=1
    cleanSignal = signal[i:(n - f)]
    # Tiene que ser un número de muestras par para ser procesada por la FFT
    if len(cleanSignal) %2 != 0 :
        cleanSignal = np.append(cleanSignal, [0])
    return cleanSignal


# /////////////////////////////////////////////////////////
# //////////////// PRE - PROCESAMIENTO ///////////////////
# ///////////////////////////////////////////////////////
def preprocess (signal):
    # Quitamos la basura y el ruido
    signal = clearSignal(signal)
    # Numero total de muestras
    mt = signal.size
    print("NUMERO DE MUESTRAS EN EL TIEMPO : ", mt)
    # Frecuencia de muestreo (el doble de la frecuencia máxima audible)
    # Por estandar es 44100
    fs = 44100.0

    # Deminimos numero de cracteristicas a persistir
    C = 20
    Y, frq = calculateSpectrumWithHPS(signal, fs)
    print("TERMINA LA FFT ")
    # Hacemos lista de (decibeles(Y), frecuencia(x)) tuples
    esp_frecuencia_pairs = [(Y[i],frq[i]) for i in range(len(Y))]
    # Ordenamos
    esp_frecuencia_pairs.sort()
    esp_frecuencia_pairs.reverse()

    return getMaximumFreq(C, esp_frecuencia_pairs)

# ///////////////////////////////////////////////////////
# //////////// PROCESAMIENTO Y CLASIFICACION ////////////
# ///////////////////////////////////////////////////////

def process(signal):
    signal = preprocess(signal)
    print("/////////// FFT /////////// \n{}".format(signal))
    #np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/fft_rafa_1.txt", signal)
    # //////////// LEEMOS TODOS LOS ELEMENTOS TRANSFORMADOS POR EL PCA /////////////
    df = pd.read_csv(
        filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_pca.csv',
        header = None,
        sep=',')

    df.dropna(how="all", inplace=True) # drops the empty line at file-end
    ancho = len(df.columns)
    Xtr = df.ix[:,0:ancho - 2].values
    Ytr = df.ix[:, ancho - 1 ].values

    # ////////// LEEMOS TODOS LOS ELEMENTOS TRANSFORMADOS POR EL PCA /////////////
    df = pd.read_csv(
        filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/w_pca.csv',
        header = None,
        sep=',')

    df.dropna(how="all", inplace=True) # drops the empty line at file-end
    ancho = len(df.columns)
    W = df.ix[:,0:ancho - 1].values

    #print ('/////////// MATRIZ DE PESOS PCA /////////\n{}'.format(W))

    #print ('/////////// FRECUENCIAS FUNDAMENTALES /////////')
    Xte = np.array(signal).reshape(-1, 1)
    #print(Xte)
    # Normalizamos datos
    Xte = StandardScaler().fit_transform(Xte).reshape(1, -1)
    #print("/////////// NORMALIZADAS \n{}".format(Xte))

    # Reducimos dimensiones
    Xte = Xte.dot(W)
    print("/////////// W * XTE /////////// \n{}".format(Xte))

    # tf Graph Input
    xtr = tf.placeholder("float")
    xte = tf.placeholder("float")

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    diference = tf.add(xtr, tf.negative(xte))
    abs_diference =  tf.abs(diference)
    distance = tf.reduce_sum(abs_diference, reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)

    accuracy = 0.

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Get nearest neighbor
        #diference_r = sess.run(diference, feed_dict={xtr: Xtr, xte: Xte})
        #abs_diference_r = sess.run(abs_diference, feed_dict={xtr: Xtr, xte: Xte})
        #distance_r = sess.run(distance, feed_dict={xtr: Xtr, xte: Xte})
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte})
        # Get nearest neighbor class label and compare it to its true label
        #print("SUMA:\n", diference_r)
        #print("ABSOLUTO:\n", abs_diference_r)
        #print("DISTANCE:\n", distance_r)
        print("nn_index", nn_index)

        print("Prediction:", Ytr[nn_index])
        return Ytr[nn_index]
        print("Done!")

#////////////// MAIN //////////////////
inputFileName = '/Users/alancruz/Documents/mujeres/m42.wav'
spf = wave.open(inputFileName,'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'uint8')
#np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/wave_open2_1.txt", signal)

process(signal)
