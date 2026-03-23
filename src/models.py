import numpy as np
import pandas as pd

class LinearRegression:
	def __init__(self, X, y, nombres_features, L1 = 0, L2 = 0):
		self.X = np.column_stack((np.ones(X.shape[0]), X))
		self.y = y
		self.nombres_features = nombres_features
		self.L1 = L1
		self.L2 = L2
	
	def entrenar_pseudo_inv(self):
		self.w = np.linalg.pinv((self.X).T @ self.X) @ (self.X).T @ self.y 
		return self.w
	
	def entrenar_gradiente_descendiente(self,alpha,iters):
		self.w = np.zeros((self.X).shape[1])
		n = (self.X).shape[0]

		for _ in range (iters):
			y_pred = self.X @ self.w
			grad = (2/n)*(self.X).T @ (y_pred - self.y)
			self.w = self.w - alpha*grad
		
		return self.w
	
	def entrenar_ridge_regression

	def entrenar_LASSO
	
	def coefs_con_features(self):
		lista_noms = self.nombres_features.tolist()
		print(round(self.w[0],4))
		for i in range((self.w).shape[0]):
			if i != 0:
				print(f' {round(self.w[i],4)} x {lista_noms[i-1]}')