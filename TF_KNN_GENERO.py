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

# //////////// LEEMOS TODOS LOS ELEMENTOS TRANSFORMADOS POR EL PCA /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_pca_knn.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
Xtr = df.ix[:,0:ancho - 2].values
Ytr = df.ix[:, ancho - 1 ].values

# ////////// LEEMOS LA MATRIZ DE PESOS DEL PCA  /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/w_pca_knn.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
W = df.ix[:,0:ancho - 1].values

print ("/////////// W ///////////\n{}".format(W))

# ////////// LEEMOS LOS DATOS DE PRUEBA /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_t.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)

Xte = df.ix[:,0:ancho - 2].values
# Normalizamos datos
#Xte = StandardScaler().fit_transform(Xte)
print("/////////// XTE ///////////\n{}".format(Xte))
Yte = df.ix[:, ancho - 1 ].values
# Reducimos dimensiones
Xte = Xte.dot(W)
print("/////////// XTE * W //////////\n{}".format(Xte))
np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/pca_data_t.csv", Xte, delimiter=",")
# tf Graph Input
xtr = tf.placeholder("float")
xte = tf.placeholder("float")

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
hombres = 0
mujeres = 0
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("nn_index", nn_index)

        print("Test", i, "Prediction:", Ytr[nn_index], "\t----\tReal Value:", Yte[i])
        # Calculate accuracy
        if Ytr[nn_index] == Yte[i]:
            if Ytr[nn_index].strip() == 'M':
                hombres+=1
            else :
                mujeres+=1
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
    print("correct M :", hombres)
    print("correct F :", mujeres)
