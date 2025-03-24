import numpy as np
import pandas as pd
import sklearn
#import surprise
from surprise import Dataset, Reader, SVD
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# Question 1:
# G = 11
# R1 = 2, R2 = 9, R3 = 6
# C1 = 2, C2 = 2, C3 = 6

file = "ratingmatrix_BDA.xlsx"
matrix = pd.read_excel(file, header=0, index_col=0)
ratings_matrix = matrix.fillna(0).to_numpy()
# Q2
print("Ratings Matrix:")
print(ratings_matrix)

svd = TruncatedSVD(n_components=3)
U = svd.fit_transform(ratings_matrix)
sigma = np.diag(svd.singular_values_)
V = svd.components_

# Q5 
# Matrix U is the user feature matrix that represents how each strongly each user(rows) relates to the latent factors
print("U Matrix: \n", U)
# Matrix Sigma is a diagnol matrix that represents the importance of each latent factor. 
# The larger the singular value, the more influential that latent factor is in explaining the variability or structure in the ratings matrix.
print("\n Sigma: \n", sigma)
# The V matrix tells us how each movie relates to the latent factors
print("\n V Matrix: \n" , V)

# Q6
# Approximatation of the ratings matrix
approx_ratingMatrix = np.dot(U, np.dot(sigma, V))
print("\nApproximate Ratings Matrix: \n", approx_ratingMatrix)

# Q7
# No vacant cells only entries with values
known_ratings_mask = ratings_matrix > 0

# Calculate the RMSE considering only known ratings and no vacant cells
rmse = np.sqrt(mean_squared_error(ratings_matrix[known_ratings_mask], approx_ratingMatrix[known_ratings_mask]))
print("\nRoot Mean Squared Error (RMSE) between original and approximate matrix: ", rmse)

# Q8
predicted_df = pd.DataFrame(approx_ratingMatrix, index=matrix.index, columns=matrix.columns)
# Create a DataFrame for the mask.
mask_df = pd.DataFrame(known_ratings_mask, index=matrix.index, columns=matrix.columns)

# Define a function to apply styles:
# - Known ratings: bold and blue.
# - Predicted ratings: italic and red.
def style_cell(val, is_known):
    if is_known:
        return 'font-weight: bold; color: blue;'
    else:
        return 'font-style: italic; color: red;'

# Apply the style to each cell based on the mask.
def style_row(row):
    # row.name gives the index label, use it to retrieve the corresponding mask row.
    mask_row = mask_df.loc[row.name]
    return [style_cell(val, is_known) for val, is_known in zip(row, mask_row)]

styled_df = predicted_df.style.apply(style_row, axis=1)


print(styled_df)

# Q13
# a. 
# b. 
# c. 
# d. 
# e. 
# f. 