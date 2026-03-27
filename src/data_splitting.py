import numpy as np
import pandas as pd
from src.models import LinearRegression
from src.metrics import ECM

def train_val_split(dev,rand_state = None):
	dev_shuffled = dev.sample(frac = 1, random_state = rand_state) #Uso random_state para que siempre se mezcle igual y comparar resultados
	total_filas = dev_shuffled.shape[0]
	q_filas_train = int(0.8*total_filas)
	train_set = dev_shuffled[:q_filas_train]
	validation_set = dev_shuffled[q_filas_train:]

	train_set.to_csv("../data/casas_train.csv", index=False)
	validation_set.to_csv("../data/casas_validation.csv", index=False)

	return train_set,validation_set

def cross_val(X,y,K,l, tipo, alfa, iters,nombres_features):
	n = len(y)
	indices = np.arange(n)
	np.random.shuffle(indices) #hago shuffle para que los datos esten variados (cuando los agarres)
	folds = np.array_split(indices,K)

	ECMs = []

	for i in range (K):
		val_idx = folds[i] #indice de la Ki parte 
		train_idx = np.concatenate([folds[j] for j in range(K) if j != i]) #lo que sobra es del train para esa iteracion

		X_train_fold = X[train_idx]
		y_train_fold = y[train_idx]
		X_val_fold = X[val_idx]
		y_val_fold = y[val_idx]

		if tipo == "L1":
			modelo_train = LinearRegression(X_train_fold,y_train_fold,nombres_features,l)
			modelo_train.entrenar_gradiente_descendiente(alfa,iters)
			
		else:
			modelo_train = LinearRegression(X_train_fold,y_train_fold,nombres_features,l)
			modelo_train.entrenar_pseudo_inv()
		
		X_val_c_bias = np.column_stack((np.ones(len(X_val_fold)), X_val_fold))
		y_pred_val = X_val_c_bias @ modelo_train.w
		ECM_i = ECM(y_val_fold,y_pred_val)

		ECMs.append(ECM_i)
	
	return np.mean(ECMs)

	