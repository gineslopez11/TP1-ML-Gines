import numpy as np
import pandas as pd

class LinearRegression:
	def __init__(self, X, y):
		self.X = np.column_stack((np.ones(X.shape[0]), X))
		self.y = y
	
	def entrenar_pseudo_inv(self):
		self.w = np.linalg.inv((self.X).T @ self.X) @ (self.X).T @ self.y 
		return self.w
	
	def entrenar_gradiente_descendiente(self,alpha,iters):
		self.w = np.zeros((self.X).shape[1])
		n = (self.X).shape[0]

		for _ in range (iters):
			y_pred = self.X @ self.w
			grad = (2/n)*(self.X).T @ (y_pred - self.y)
			self.w = self.w - alpha*grad
		
		return self.w