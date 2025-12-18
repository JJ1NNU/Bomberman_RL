import sys
import os
import pickle
import numpy as np

# seems to work as expected
def merge(matrices):

    mats = len(matrices)
    cols = len(matrices[0])
    rows = len(matrices[0][0])

    print(f"mats {mats}")
    print(f"cols {cols}")
    print(f"rows {rows}")

    sum = np.zeros((cols, rows))
    for i in range(mats):
        sum += matrices[i]
    mean_matrix = sum / mats

    print(type(matrices))
    print(type(matrices[0]))
    print(type(matrices[0][0]))
    print(type(matrices[0][0][0]))
    
    return mean_matrix
    
matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix2 = np.array([[7, 6, 5], [4, 3, 2]])
matrix3 = np.array([[1, 1, 1], [1, 1, 1]])
matrix4 = np.array([[1, 1, 1], [1, 1, 1]])
matrix5 = np.array([[0, 0, 0], [0, 0, 0]])
matrices = [matrix1, matrix2, matrix3, matrix4, matrix5]

result = merge(matrices)
print(result) # should print [[2. 2. 2.],[2. 2. 2.]] 


# Erstelle eine Liste von Arrays mit der gewünschten Form und fülle sie mit Einsen
arrays = [np.ones((790128, 5)) * (i+1) for i in range(10)]
# 1+2+3+4+5+6+7+8+9+10 = 55
# 55 / 10 = 5.5


result_two = merge(arrays)
print(result_two)   # should print 10 rows of 5.5s
