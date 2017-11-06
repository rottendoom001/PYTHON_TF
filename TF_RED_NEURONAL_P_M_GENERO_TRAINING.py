import pandas as pd
import numpy as np
import tensorflow as tf

# //////////// LEEMOS TODOS LOS ELEMENTOS TRANSFORMADOS POR EL PCA /////////////
df = pd.read_csv(
    filepath_or_buffer='/Users/alancruz/Desktop/PYTHON/CORE/data/result_pca.csv',
    header = None,
    sep=',')

n = 10  # Numero de característias de entrada
m = 1   # Numero de Salidas
h = 4   # Numero de neuronas en la capa oculta


df.dropna(how="all", inplace=True) # drops the empty line at file-end
ancho = len(df.columns)
Xtr = df.ix[:,0:ancho - 2].values
Ytr = df.ix[:, ancho - 1 ].values
print ("X >", Xtr.shape)
print ("Y >", Ytr.shape)
k = 0

while (k < len(Ytr)) :
    if Ytr[k].strip() == 'M':
        Ytr[k] = 0
    else :
        Ytr[k] = 1
    k+=1
Ytr = Ytr.reshape((len(Xtr), m))
print ("Y >", Ytr.shape)

x_ = tf.placeholder(tf.float32, shape=[len(Xtr), n], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[len(Xtr), m], name="y-input")

W1 = tf.Variable(tf.random_uniform([n,h], -1, 1), name="W1")
B1 = tf.Variable(tf.zeros([h]), name="Bias1")
output_1 = tf.sigmoid(tf.matmul(x_, W1) + B1)

W2 = tf.Variable(tf.random_uniform([h, m], -1, 1), name="W2")
B2 = tf.Variable(tf.zeros([m]), name="Bias2")
PRED = tf.sigmoid(tf.matmul(output_1, W2) + B2)

cost = tf.reduce_mean(( (y_ * tf.log(PRED)) +
        ((1 - y_) * tf.log(1.0 - PRED)) ) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(400000):
        _, c = sess.run([train_step, cost], feed_dict={x_: Xtr, y_: Ytr})
        if c < 0.3 :
            print('Pred :\n', sess.run(PRED, feed_dict={x_: Xtr, y_: Ytr}))
            print('W1 :\n', sess.run(W1))
            print('B1 :\n', sess.run(B1))
            print('W2 :\n', sess.run(W2))
            print('B2 :\n', sess.run(B2))
            print('Final Cost :', c)
            break
        if i % 1000 == 0:
            print('/////////////////////////////')
            print('Epoch ', i)
            print('cost :', c)
