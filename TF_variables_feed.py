import tensorflow as tf

grafo1=tf.Graph()
with grafo1.as_default():
    # Variable que guardará el sumatorio en las distintas ejecuciones
    sumatorio = tf.Variable(0, name="counter")
    # Variable de entrada (feed) = contador
    input1 = tf.placeholder(tf.int32)

    # Logica de cada paso
    valor_operacion = tf.multiply(input1, input1)
    flujo=tf.assign_add(sumatorio, valor_operacion)


with tf.Session(graph=grafo1) as sess:
    # Las variables deben inicializarse explicitamente
    sess.run(tf.global_variables_initializer())

    for contador in range(4):
        estado= sess.run([flujo,valor_operacion],feed_dict={input1:contador})
        print ("Iteracion {}: {}".format(contador,estado))
