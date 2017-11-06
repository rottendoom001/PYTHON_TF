import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 1

# Network Parameters
n_hidden_1 = 1 # Numero de neuronas en capa oculta
n_input = 2 # Nuero de entradas
n_classes = 1 # numero de salidas
display_step = 1

datos_x = np.array([[0, 0]
                 ,[1, 0]
                 ,[0, 1]
                 ,[1, 1]])

datos_y = np.array([[0],[1],[1],[0]])

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def binary_activation(x):
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out

def simple_optimizer(e, weights, biases):
    

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = binary_activation(layer_1)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    out_layer = binary_activation(out_layer)
    return out_layer

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(datos_y - pred ))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c, w1, w2, b1, b2, pred_f = sess.run([
                        optimizer,
                        cost,
                        weights['h1'],
                        weights['out'],
                        biases['b1'],
                        biases['out'],
                        pred]
                        , feed_dict={x: datos_x, y: datos_y})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c))
    print("Optimization Finished!")
    print ("w1 :\n{}".format(w1))
    print ("b1 :\n{}".format(b1))
    print ("w2 :\n{}".format(w2))
    print ("b2 :\n{}".format(b2))
    print ("pred :\n{}".format(pred_f))
