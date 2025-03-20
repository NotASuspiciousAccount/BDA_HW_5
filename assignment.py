import numpy as np
import pandas as pd
import sklearn

from sklearn.decomposition import TruncatedSVD

file = "ratingmatrix_BDA.xlsx"
importmatrix = pd.read_excel(file, header=0, index_col=0)
ratings_matrix = importmatrix.fillna(0).to_numpy()

# Question 1:
# G = 11
# R1 = 2, R2 = 9, R3 = 6
# C1 = 2, C2 = 2, C3 = 6

svd = TruncatedSVD(n_components=3)
U = svd.fit_transform(ratings_matrix)
sigma = np.diag(svd.singular_values_)
V = svd.components_

# Q5 
# Matrix U is essentially 
print("U Matrix: \n", U)
print("\n Sigma: \n", sigma)
print("\n V Matrix: \n" , V)