import pandas as pd
import numpy as np
import tensorflow as tf

def printNvalues(n, arr) :
    i = 0
    for v in arr :
        print ("[{}] = {}".format(i,v))
        if n == i :
            break
        i+=1

df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/TENSOR_FLOW/data/cancer.csv',
    header = 0,
    sep=',')

df.dropna(how="all", inplace=True) # drops the empty line at file-end


X = df.ix[:,2:].values
Y = df.ix[:,1].values

printNvalues(0,X)
printNvalues(0,Y)

datos_prueba = 50

# In this example, we limit mnist data
Xtr = X[0:(len(X) - datos_prueba ),:]
Ytr = Y[0:(len(Y) - datos_prueba )]

print (len(Xtr))
print (len(Ytr))

Xte = X[(len(X) - datos_prueba ):len(X),:]
Yte = Y[(len(Y) - datos_prueba ):len(Y)]

print (len(Xte))
print (len(Yte))

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
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("nn_index", nn_index)

        print("Test", i, "Prediction:", Ytr[nn_index], \
            "True Class:", Yte[i])
        # Calculate accuracy
        if Ytr[nn_index] == Yte[i]:
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
