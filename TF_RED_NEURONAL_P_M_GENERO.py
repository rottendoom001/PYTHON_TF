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

# ////////// LEEMOS LA MATRIZ DE PESOS DEL PCA  /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/w_pca.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
W = df.ix[:,0:ancho - 1].values

#print ("/////////// W ///////////\n{}".format(W))

# ////////// LEEMOS LOS DATOS DE PRUEBA /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_t.csv',
    header = None,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)

Xte = df.ix[:,0:ancho - 2].values

#print("/////////// XTE ///////////\n{}".format(Xte))
Yte = df.ix[:, ancho - 1 ].values

# //// DATOS DE LA RED NEURONAL ///////
n = 10  # Numero de característias de entrada
m = 1   # Numero de Salidas
h = 4   # Numero de neuronas en la capa oculta

k = 0
while (k < len(Yte)) :
    if Yte[k].strip() == 'M':
        Yte[k] = 0
    else :
        Yte[k] = 1
    k+=1
Yte = Yte.reshape((len(Xte), m))
print ("Y >", Yte)

# Reducimos dimensiones
Xte = Xte.dot(W)
#print("/////////// XTE * W //////////\n{}".format(Xte))

x_ = tf.placeholder(tf.float32, shape=[1 , n], name="x-input")

W1 = tf.constant([[ -2.27649403e+00,  -6.63474023e-01,   1.57575834e+00,  -3.55812609e-01],
 [ -1.66783297e+00,  -1.25870132e+00,   1.38324976e+00,  -2.87358370e-03],
 [ -1.00611603e+00,   3.01318377e-01,   5.48789680e-01,  -5.13256252e-01],
 [  7.09671259e-01,  -1.58599508e+00,   3.17886591e-01,   2.42182159e+00],
 [  1.47929537e+00,   2.05749536e+00,   8.49874318e-01,   1.72604191e+00],
 [  8.79072547e-01,   2.17223215e+00,   3.77681613e+00,  -1.66010642e+00],
 [ -9.65735316e-01,  -2.16943526e+00,  -1.19214416e+00,  -2.45818645e-01],
 [ -6.18004441e-01,  -6.39787614e-01,  -8.17957819e-01,  -2.49989700e+00],
 [ -1.16214208e-01,   3.05116820e+00,   7.57704452e-02,   2.53080535e+00],
 [  9.16602552e-01,   3.07038808e+00,   2.89418507e+00,   1.61316490e+00]], shape=[n,h], name="W1")
B1 = tf.constant([-1.18635821,  0.50047314,  0.86380291, -0.62256706], shape=[h], name="Bias1")
output_1 = tf.sigmoid(tf.matmul(x_, W1) + B1)

W2 = tf.constant([[ 4.06563807],
 [-5.3782053 ],
 [ 3.84682631],
 [ 4.67790556]], shape=[h, m], name="W2")
B2 = tf.constant([-2.77817512], shape=[m], name="Bias2")
PRED = tf.sigmoid(tf.matmul(output_1, W2) + B2)
#/////////////////////////////////

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
        x = Xte[i, :].reshape(1,n)
        result = sess.run(PRED, feed_dict={x_: x})
        # Get nearest neighbor class label and compare it to its true label
        print("result:", result)
'''
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
'''
