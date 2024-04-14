'''
Author: 27
LastEditors: 27
Date: 2024-04-14 11:56:34
LastEditTime: 2024-04-14 11:56:52
FilePath: /Yume-MBA-homework/yumi/src/draw.py
description: type some description
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def plot_decision_boundary(model: LogisticRegression, axis):
    
    # x0, x1, x2, x3, x4, x5= np.meshgrid(
    #     np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
    #     np.linspace(axis[0], axis[2], int((axis[1]-axis[0])*100)).reshape(-1, 1),
    #     np.linspace(axis[0], axis[3], int((axis[1]-axis[0])*100)).reshape(-1, 1),
    #     np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    #     np.linspace(axis[1], axis[2], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    #     np.linspace(axis[1], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    # )
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*200)).reshape(-1, 1),
        
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*200)).reshape(-1, 1)
    )
    # X_new = np.c_[x0.ravel(), x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel(), x5.ravel()]
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
