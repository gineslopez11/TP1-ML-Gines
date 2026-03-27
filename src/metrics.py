import numpy as np

def ECM (y_real, y_pred, L1 = 0, L2 = 0, w = 0):
	n = len(y_real)
	return (1/n)*np.sum((y_real-y_pred)**2) + L1*np.sum(np.abs(w)) + L2*np.sum(w**2)
