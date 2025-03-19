import numpy as np
import pandas as pd
import sklearn
import surprise

rating_matrix_file = "rating_matrix_BDA.xlsx"
rating_matrix = pd.read_excel(rating_matrix_file, header=0, index_col=0)

# Question 1:
# G = 11
# R1 = 2, R2 = 9, R3 = 6
# C1 = 2, C2 = 2, C3 = 6
