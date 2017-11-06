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
'''df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_pca.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
Xtr = df.ix[:,0:ancho - 2].values
Ytr = df.ix[:, ancho - 1 ].values
'''
# KMEANS CON PCA
'''
Xtr = np.array(
    [[1.4645638, 0.0089823, -0.0034565, -0.0002905, 0.0003505, -0.0062592, -0.0035117,
        0.0017938, -0.0022662, 0.0027392],
    [-4.2788236, -0.0262425, 0.0100985, 0.0008487, -0.0010240, 0.0182868, 0.0102595,
        -0.0052407, 0.0066208, -0.0080027]], dtype=np.float)
'''
# KMEANS SIN PCA
Xtr = np.array(
    [[242.9197466, 247.6436994, 249.1601627, 248.3910423, 251.7958297, 252.3851594,
      251.1527649, 254.1097679, 252.6753017, 254.3548180, 254.2330883, 255.2306830,
      259.1195428, 258.7011136, 257.2582694, 259.6271851, 259.7329793, 258.2851208,
      257.8321416, 263.2838293, 261.8883176],
    [157.4891612, 159.4217892, 160.0708088, 160.6586593, 161.4441359, 162.0742274,
     161.6816262, 163.0779285, 163.4261062, 163.5148529, 164.2220731, 164.7408591,
     165.0324992, 164.8380229, 165.2380479, 166.8780118, 166.6080751, 166.9610438,
     166.9961243, 167.5619391, 166.6455673]], dtype=np.float)

# ////////// LEEMOS LA MATRIZ DE PESOS DEL PCA  /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/w_pca.csv',
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
Xte = StandardScaler().fit_transform(Xte)
print("/////////// XTE ///////////\n{}".format(Xte))
Yte = df.ix[:, ancho - 1 ].values
# Reducimos dimensiones
#Xte = Xte.dot(W)
print("/////////// XTE * W //////////\n{}".format(Xte))
#np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/pca_data_t.csv", Xte, delimiter=",")
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
        r = 'F' if nn_index==0 else 'M'
        print("Test", i, "Prediction:", r, "\t----\tReal Value:", Yte[i])
        # Calculate accuracy
        if r.strip() == Yte[i].strip():
            if r.strip() == 'M':
                hombres+=1
            else :
                mujeres+=1
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
    print("correct M :", hombres)
    print("correct F :", mujeres)
