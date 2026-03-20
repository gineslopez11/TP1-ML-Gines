import numpy as np

def ECM (y_real, y_pred):
	n = len(y_real)
	return (1/n)*np.sum((y_real-y_pred)**2)
