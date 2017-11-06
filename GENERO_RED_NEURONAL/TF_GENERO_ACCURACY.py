import pandas as pd
import numpy as np
import tensorflow as tf
import CORE as core
import math
from datetime import datetime
from sklearn.preprocessing import StandardScaler

np.random.seed(7)
#+++++++++++++++++ ENTRENAMIENTO ++++++++++++
# // Leemos resultado del PCA para entrenar /////////////
Xtr, Ytr = core.read_training_data('/Users/alancruz/Desktop/PYTHON/CORE/data/result_pca.csv')
# // Cambiamos las Ms y Fs por 0 y 1
Ytr = core.transform_tags_to_0_1(Ytr)
# // Obtenemos modelo de red para entrenamiento
PRED, W1, B1, W2, B2, y_, x_ = core.training_model(Xtr)
#PRED, W1, B1, W2, B2, W3, B3, y_, x_ = core.training_model(Xtr)
# // Modelo para calculo del Costo
cost = tf.reduce_mean(( (y_ * tf.log(PRED)) +
        ((1 - y_) * tf.log(1.0 - PRED)) ) * -1)
# // Modelo de Optimizacion
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

#+++++++++++++++++ PRUEBAR MODELO Y SACAR ACCURACY ++++++++++++
# // Leemos datos de entrada para probar
Xte, Yte = core.read_test_data('/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_t.csv')
# // Obtenemos Matriz de pesos del PCA
PCA_W, col_number = core.get_pca_w()
# // Cambiamos las Ms y Fs por 0 y 1
Yte = core.transform_tags_to_0_1(Yte)

# Normalizamos datos
Xte = StandardScaler().fit_transform(Xte)
# Reducimos dimensiones
Xte = Xte.dot(PCA_W)
# // Obtenemos modelo de red para entrenamiento
TEST_PRED, xt_ = core.test_model(W1, B1, W2, B2)
#TEST_PRED, xt_ = core.test_model(W1, B1, W2, B2, W3, B3)
accuracy = 0.

# ++++++++++++++++++ EJECUTAMOS MODELOS ++++++++
init = tf.initialize_all_variables()
with tf.Session() as sess:
    before = datetime.now()
    sess.run(init)
    # ////////// PARA ENTRENAMOENTO //////
    #for i in range(4000000):
    i = 0
    while True:
        _, c = sess.run([train_step, cost], feed_dict={x_: Xtr, y_: Ytr})
        p = sess.run(PRED, feed_dict={x_: Xtr, y_: Ytr})
        '''or math.isnan(c) '''
        if c < 0.3:
            p = core.round_results(p)
            print('Pred :\n', p)
            np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/pred.csv", p, delimiter=",")
            r_w1 = sess.run(W1)
            r_b1 = sess.run(B1)

            r_w2 = sess.run(W2)
            r_b2 = sess.run(B2)
            '''
            r_w3 = sess.run(W3)
            r_b3 = sess.run(B3)
            '''
            print('Final Cost :', c)
            break
        if i % 1000 == 0:
            print('/////////////////////////////')
            print('Epoch ', i)
            print('cost :', c)
        # ////////// PARA ENTRENAMOENTO //////
        i+=1

    print('/////////// INICIA PUREBA DE PARA ACCURACY //////////////////')
    mujeres = 0
    hombres = 0
    for i in range(len(Xte)):
        # Get nearest neighbor
        x = Xte[i, :].reshape(1, col_number)
        result = sess.run(TEST_PRED, feed_dict={xt_: x})
        # Get nearest neighbor class label and compare it to its true label
        result = core.round_results(result)
        print("result:", int (result[0]), "correct Value:", int (Yte[i]))
        if int (result[0]) == int (Yte[i]):
            if int(result [0]) == int(0):
                hombres+=1
            else :
                mujeres+=1
            accuracy += 1./len(Xte)
    after = datetime.now()
    time = after - before
    print("accuracy:", accuracy)
    print("correct M :", hombres)
    print("correct F :", mujeres)
    print("time :", time)

    if accuracy > 0.5 :
        print ("w1", r_w1)
        print ("b1", r_b1)
        np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/w1.csv", r_w1, delimiter=",")
        np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/b1.csv", r_b1, delimiter=",")
        print ("w2", r_w2)
        print ("b2", r_b2)
        np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/w2.csv", r_w2, delimiter=",")
        np.savetxt("/Users/alancruz/Desktop/PYTHON/CORE/data/b2.csv", r_b2, delimiter=",")
        #print ("w3", r_w3)
        #print ("b3", r_b3)
