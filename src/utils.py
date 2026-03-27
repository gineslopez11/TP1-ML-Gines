import numpy as np
import pandas as pd
from src.models import LinearRegression
from src.metrics import ECM


def learning_curve(porcentajes, X_train, y_train, X_val, y_val, nombres, L2=0):
    ecm_train_list = []
    ecm_val_list = []
    
    for p in porcentajes:
        n = int(len(y_train) * p)
        X_sub = X_train[:n]
        y_sub = y_train[:n]
        
        modelo = LinearRegression(X_sub, y_sub, nombres, L2=L2)
        modelo.entrenar_pseudo_inv()
        
        y_pred_train = modelo.X @ modelo.w
        ecm_train_list.append(ECM(y_sub, y_pred_train))

        X_val_bias = np.column_stack((np.ones(len(X_val)), X_val))
        y_pred_val = X_val_bias @ modelo.w
        ecm_val_list.append(ECM(y_val, y_pred_val))
    
    return ecm_train_list, ecm_val_list
