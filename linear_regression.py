import copy, math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

X_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y_train = np.array([6, 8, 9, 11])

print(f"X_train shape: {X_train.shape}, X type: {type(X_train)}")
print(X_train)
print(f"y_train shape: {y_train.shape}, y type: {type(y_train)}")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def predict_single_loop(x, w, b):
    f_wb = 0.0
    n = x.shape[0]
    for j in range(n):
        f_wb += w[j] * x[j]
    f_wb += b
    return f_wb

x_vec = X_train[0, :]
print(f"x_vec shape: {x_vec.shape}, x_vec type: {type(x_vec)}")

# make a prediction for the first example
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape: {f_wb.shape}, prediction: {f_wb}")