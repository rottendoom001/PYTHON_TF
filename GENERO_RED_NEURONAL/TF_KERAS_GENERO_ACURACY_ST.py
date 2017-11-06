from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

import CORE as core
import numpy as np

NUM_CLASSES = 2
EPOCHS = 1000
BATCH_SIZE = 128

#/////// DATA TRAINING ///////
Xtr, Ytr = core.read_training_data('/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_st_ceps.csv')
# // Cambiamos las Ms y Fs por 0 y 1
Ytr = core.transform_tags_to_0_1(Ytr)
Ytr = np_utils.to_categorical(Ytr)
print ("Ytr : ", Ytr)

#/////// DATA TEST ///////
Xte, Yte = core.read_training_data('/Users/alancruz/Desktop/PYTHON/CORE/data/result_fft_hps_t_st_ceps.csv')
# // Cambiamos las Ms y Fs por 0 y 1
Yte = core.transform_tags_to_0_1(Yte)
Yte = np_utils.to_categorical(Yte)
print ("Yte : ", Yte)


_, n = Xtr.shape  # Numero de característias de entrada
print ("n =", n)


model = core.config_keras_model(n, NUM_CLASSES, EPOCHS)
# Entrenamos
model.fit(Xtr, Ytr, epochs=EPOCHS)

print("[INFO] Evaluating MLP model...")
(loss_mlp, accu_mlp) = model.evaluate(Xte, Yte, batch_size=BATCH_SIZE, verbose=1)

print("[INFO] loss_mlp={:.4f}, accuracy_mlp: {:.4f}%".format(loss_mlp,
                                                             accu_mlp * 100))
