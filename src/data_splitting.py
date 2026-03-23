import numpy as np
import pandas as pd

def train_val_split(dev,rand_state = None):
	dev_shuffled = dev.sample(frac = 1, random_state = rand_state) #Uso random_state para que siempre se mezcle igual y comparar resultados
	total_filas = dev_shuffled.shape[0]
	q_filas_train = int(0.8*total_filas)
	train_set = dev_shuffled[:q_filas_train]
	validation_set = dev_shuffled[q_filas_train:]

	train_set.to_csv("../data/casas_train.csv", index=False)
	validation_set.to_csv("../data/casas_validation.csv", index=False)

	return train_set,validation_set


	