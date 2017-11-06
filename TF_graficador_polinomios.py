import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

def build_x_vector(x, grade):
    x_new = np.zeros([x.size, grade])
    for i in range(grade):
        x_new[:,i] = np.power(x,(i+1))
    return x_new

def build_w_vector(grade):
    w = np.zeros((grade,1))
    for i in range(grade):
        w[i,:] = random.randrange(1, 100)
    return w

plt.ion()
fig, ax = plt.subplots(1, 1)

#s = [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13, 14, 15, 16, 17]
xs = range(-10, 10)
xs = np.array(xs)

n = 8
W = tf.Variable(build_w_vector(n), dtype=tf.float32)
B = tf.Variable([-50000000000.], dtype=tf.float32)
X = tf.placeholder(tf.float32)

X_new = build_x_vector(xs, n)
print("X_new : \n%s"%(X_new))
Y_pred = tf.add(tf.matmul(X,W),B)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y, w , b = sess.run([Y_pred, W, B], feed_dict={X: X_new})
    print("Y = %s"%(y))
    print("W : \n%s"%(w))
    print("b : \n%s"%(b))

    ax.scatter(xs, y)
    fig.show()
    plt.draw()
plt.waitforbuttonpress()
