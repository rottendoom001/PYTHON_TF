"""Simple tutorial for using TensorFlow to compute polynomial regression.
Parag K. Mital, Jan. 2016"""
# %% Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Normalizar datos
def normalizar (xs):
    ln = []
    maxValue = max(xs)
    minValue = min(xs)

    for e in xs :
        # Formula para nomalizar
        # (x - min(x)) / (max(x) - min(x))
        r = ( e - minValue ) / ( maxValue - minValue )
        ln.append(r)
    return ln


# %% Let's create some toy data
plt.ion()
n_observations = 6
fig, ax = plt.subplots(1, 1)
#xs = np.linspace(-3, 3, n_observations)
#ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
xs = [1,2,3,4,5,6]
ys = [449,525,412,161,639,732]

ysn = normalizar(ys)
print (ysn)

ax.scatter(xs, ysn)
fig.show()
plt.draw()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# %% Instead of a single factor and a bias, we'll create a polynomial function
# of different polynomial degrees.  We will then learn the influence that each
# degree of the input (X^0, X^1, X^2, ...) has on the final output (Y).
Y_ini = tf.Variable(tf.random_normal([1]), name='bias')
print ('/////////////')
Y_pred = Y_ini
for pow_i in range(1, 5):
    print("Potencia : %s "%(pow_i))
    '''
    if pow_i == 1 :
        W1 = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    if pow_i == 2 :
        W2 = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    if pow_i == 3 :
        W3 = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    if pow_i == 4 :
        W4 = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    ''''
    '''
    PW = tf.pow(X, pow_i)
    MUL = tf.multiply(PW, W)
    Y_pred = tf.add(MUL, Y_pred)
    '''
    Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

print ('/////////////')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_ini = sess.run(Y_ini)
    print ("Y_ini = [{}]".format(y_ini))
    for x in xs:
        #w1, w2, w3, w4 = sess.run([W1, W2, W3, W4], feed_dict={X: x})
        #print (" W1[{}] -- W2[{}] -- W3[{}] -- W4[{}]".format(w1, w2, w3, w4))
        #w, pw, mul, res, y_ini = sess.run([W, PW, MUL, Y_pred, Y_ini], feed_dict={X: x})
        #print (" W[{}] -- PW[{}] -- MUL[{}] -- Y[{}] -- Y_ini = [{}] ".format(w, pw, mul, res, y_ini))

plt.waitforbuttonpress()
