import pandas as pd 
import numpy as np 

file_path = 'cv_3.xlsx'

sheet_data = pd.read_excel(file_path, sheet_name='Sheet1')

A = sheet_data.to_numpy()
print(A)

def find_homography(A):
    _, _, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    return L

P = find_homography(A)
print(P)
