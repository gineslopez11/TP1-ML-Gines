import numpy as np
import pandas as pd

def crear_features(DataF):                                                                                                                                            
	DataF['metros_cubiertos / Área'] = DataF['metros_cubiertos'] / DataF['Área'].replace(0, np.nan)                                                                   
	DataF['ambientes x Ubicacion'] = DataF['ambientes'] * DataF['Ubicacion']
	DataF['metros_cubiertos / ambientes'] = DataF['metros_cubiertos'] / DataF['ambientes'].replace(0, np.nan)  
	
def crear_features_polinomicas(DataF,columnas,grado_max): #generan features como las del ejemplo de la consigna
	for col in columnas:
		for grado in range(2, grado_max + 1):
			DataF[f'{col}^{grado}'] = DataF[col]**grado
	