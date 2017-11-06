
# %% Imports
import numpy as np
import tensorflow as tf
import pandas as pd
# Neurona con TensorFlow
# Defino las entradas
entradas = tf.placeholder("float", name='Entradas')
datos = np.array([[0, 0]
                 ,[1, 0]
                 ,[0, 1]
                 ,[1, 1]])

# Definiendo pesos y sesgo
pesos = tf.placeholder("float", name='Pesos')
sesgo = tf.placeholder("float", name='Sesgo')

# Defino las salidas
uno = lambda: tf.constant(1.0)
cero = lambda: tf.constant(0.0)

# Función de activación
activacion = tf.reduce_sum(tf.add(tf.matmul(entradas, pesos), sesgo))
neurona = tf.case([(tf.less(activacion, 0.0), cero)], default=uno)



with tf.Session() as sess:
    # para armar tabla de verdad
    x_1 = []
    x_2 = []
    out = []
    act = []
    for i in range(len(datos)):
        print ("//////////////")
        print ("datos :\n{}".format(datos[i]))
        t = datos[i].reshape(1, 2)
        print ("t :\n{}".format(t))
        p = np.array([[1.],[1.]])
        print ("pesos iniciales :\n{}".format(p))

        salida, activ, W = sess.run([neurona, activacion, pesos], feed_dict={entradas: t,
                                        pesos: p,
                                        sesgo: -1.5})
        print ("pesos finales :\n{}".format(W))
        # armar tabla de verdad en DataFrame
        x_1.append(t[0][0])
        x_2.append(t[0][1])
        out.append(salida)
        act.append(activ)
    tabla_info = np.array([x_1, x_2, act, out]).transpose()
    tabla = pd.DataFrame(tabla_info,
                         columns=['x1', 'x2', 'f(x)', 'x1 AND x2'])
print(tabla)
