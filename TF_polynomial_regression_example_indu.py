
# %% Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def modify_input(x, grade):
    x_new = np.zeros([x.size, grade])
    for i in range(grade):
        x_new[:,i] = np.power(x,(i+1))
        x_new[:,i] = x_new[:,i]/np.max(x_new[:,i])
    return x_new

plt.ion()
fig, ax = plt.subplots(1, 1)

x_input = np.linspace(0,3,1000)
#print ("x_input: %s"%(x_input))
x1 = x_input/np.max(x_input)
#print ("x1: %s"%(x1))
x2 = np.power(x_input,2)/np.max(np.power(x_input,2))
#print ("x2: %s"%(x2))
y_input = 5*x1-3*x2
y_input = y_input.reshape((y_input.size,1))

n = 2
W = tf.Variable(tf.random_normal([n,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None,n])
Y = tf.placeholder(tf.float32, shape=[None,1])


X_new = modify_input(x_input, n)

#print("X_new : \n%s"%(X_new))
Y_pred = tf.add(tf.matmul(X,W),b)
loss = tf.reduce_mean(tf.square(Y_pred - Y))

learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

n_epochs = 5000
with tf.Session() as sess:
    for step in range(n_epochs):
        sess.run(tf.global_variables_initializer())
        _, c = sess.run([optimizer, loss], feed_dict={X: X_new, Y: y_input})
        if step % 100 == 0:
            print("%s) COSTO: %s"%(step,c))
        if (c < 0.0001) :
            break


    print ("W: %s"%(sess.run(W)))
    print ("b: %s"%(sess.run(b)))

    y_test = sess.run(Y_pred, feed_dict={X: X_new})
    #ax.scatter(x_input, y_input)
plt.plot(x_input,y_input)
plt.plot(x_input,y_test)
#fig.show()
plt.show()
plt.waitforbuttonpress()
