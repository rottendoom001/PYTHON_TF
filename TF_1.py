import tensorflow as tf
sess = tf.Session()

#//////////// EJEMPLO 1 //////////////
#node1 = tf.constant(3.0, dtype=tf.float32)
#node2 = tf.constant(4.0) # also tf.float32 implicitly
#print (node1, node2)


#sess = tf.Session()
#print(sess.run([node1, node2]))

#//////////// EJEMPLO 2 //////////////
#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#adder_node = a + b
#print(sess.run(adder_node, {a: 3, b:4.5}))
#print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

#add_and_triple = adder_node * 3.
#print(sess.run(add_and_triple, {a: 3, b:4.5}))

#//////////// EJEMPLO 3 - linear - regression//////////////
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# Para inicializar las variables que sólo se dejaron, en este caso x
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print("FACTOR DE ERROR:", sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Si queremos corregir de forma manual
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print("FACTOR DE ERROR:", sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Si queremos que se corrija automáticmente
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))
