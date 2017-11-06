"""Simple tutorial for using TensorFlow to compute polynomial regression.
Parag K. Mital, Jan. 2016"""
# %% Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Normalizar datos
def normalize (xs):
    ln = []
    maxValue = max(xs)
    minValue = min(xs)

    for e in xs :
        # Formula para nomalizar
        # (x - min(x)) / (max(x) - min(x))
        r = ( e - minValue ) / ( maxValue - minValue )
        ln.append(r)
    return ln

# Construir el vector de valores para X

def build_x_vector(x, grade):
    x_new = np.zeros([x.size, grade])
    for i in range(grade):
        x_new[:,i] = np.power(x,(i+1))
    return x_new

# %% Let's create some toy data
plt.ion()
fig, ax = plt.subplots(1, 1)
#xs = np.linspace(-3, 3, n_observations)
#ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
xs = [1,2,3,4,5,6,7,8,9,10,11,12,13]
xs = np.array(xs)
#ys = [449,525,412,161,639,732]
ysn = [15, 14, 13, 6, 19, 16, 16, 11, 17, 15, 12, 25, 14]
#ysn = normalize(ys)
print (ysn)

ax.scatter(xs, ysn)
fig.show()
plt.draw()

# Grado de la ecuación polinomial
n = 15
W = tf.Variable(tf.random_normal([n,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

X_new = build_x_vector(xs, n)
print("X_new : \n%s"%(X_new))
Y_pred = tf.add(tf.matmul(X,W),b)

loss = tf.reduce_mean(tf.square(Y_pred - Y))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

n_epochs = 1000
with tf.Session() as sess:
    for step in range(n_epochs):
        sess.run(tf.global_variables_initializer())
        _, c = sess.run([optimizer, loss], feed_dict={X: X_new, Y: ysn})
        if step % 10 == 0:
            print("COSTO: %s"%(c))

    print ("W: %s"%(sess.run(W)))
    print ("b: %s"%(sess.run(b)))

    y_test = sess.run(Y_pred, feed_dict={X: X_new})
    ax.scatter(xs, y_test)
    fig.show()
    plt.draw()

plt.waitforbuttonpress()
