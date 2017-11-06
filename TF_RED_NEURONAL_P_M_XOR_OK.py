import numpy as np
import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

W1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="W1")
W2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="W2")

B1 = tf.Variable(tf.zeros([2]), name="Bias1")
B2 = tf.Variable(tf.zeros([1]), name="Bias2")

A1 = tf.sigmoid(tf.matmul(x_, W1) + B1)
PRED = tf.sigmoid(tf.matmul(A1, W2) + B2)

cost = tf.reduce_mean(( (y_ * tf.log(PRED)) +
        ((1 - y_) * tf.log(1.0 - PRED)) ) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
print (XOR_X.shape)
XOR_Y = [[0],[1],[1],[0]]

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    for i in range(100000):
        sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

        if i % 1000 == 0:
            print('/////////////////////////////')
            print('Epoch ', i)
            print('Pred :\n', sess.run(PRED, feed_dict={x_: XOR_X, y_: XOR_Y}))
            print('W1 :\n', sess.run(W1))
            print('B1 :\n', sess.run(B1))
            print('W2 :\n', sess.run(W2))
            print('B2 :\n', sess.run(B2))
            print('cost :', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
