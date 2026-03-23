import numpy as np
import pandas as pd

def normalizar (columnas,train,validation = None): #si se desea normalizar todo el dev, se agrega la parte de valå
	train_normalizado = train.copy()
	media_var = {}
	for col in columnas:
			media_var[col] = (train_normalizado[col].mean(), train_normalizado[col].var())

	#Consigo media y varianza para poder normalizar
	#--> Formula de normalizacion a usar = (x - media) / desvio

	if validation is not None:
		validation_normalizado = validation.copy()
		for parte in [train_normalizado,validation_normalizado]:
			for col in columnas:
					media,var = media_var[col]
					parte[col] = (parte[col] - media) / np.sqrt(var)
		
		return [train_normalizado,validation_normalizado]
	
	else:
		for col in columnas:
			media,var = media_var[col]
			train_normalizado[col] = (train_normalizado[col] - media) / np.sqrt(var)
		return train_normalizado