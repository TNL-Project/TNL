import numpy as np
import csv
import os

def export_matrix_to_csv(matrix, filename, subfolder='CSV_DenseMatrices'):
    # Ensure the subfolder exists
    os.makedirs(subfolder, exist_ok=True)

    # Modify the filename to include the subfolder path
    filename_with_path = os.path.join(subfolder, filename)

    with open(filename_with_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix:
            writer.writerow(['{:.15f}'.format(val) for val in row])

# Parameters
n_rows1 = 100
n_columns = 150 # NumColumns1 == NumRows2
n_columns2 = 110

h_x = 1.0 / 100
h_y = 1.0 / 100

# Initialize matrices
matrix1 = np.zeros((n_rows1, n_columns))
matrix2 = np.zeros((n_columns, n_columns2))

# Fill matrix1
for i in range(n_rows1):
    for j in range(n_columns):
        matrix1[i, j] = np.sin(2 * np.pi * h_x * i) + np.cos(2 * np.pi * h_y * j)

# Fill matrix2
for i in range(n_columns):
    for j in range(n_columns2):
        matrix2[i, j] = np.sin(2 * np.pi * h_x * i) + np.cos(2 * np.pi * h_y * j)

# Calculate the matrix product
checkMatrix = np.dot(matrix1, matrix2)

# Transpose matrices before exporting
matrix1_transposed = np.transpose(matrix1)
matrix2_transposed = np.transpose(matrix2)

# For the Inplace Transposition
n_square = 110

# Initialize matrix
matrixInPlace = np.zeros((n_square, n_square))

# Fill matrixInPlace
for i in range(n_square):
    for j in range(n_square):
        matrixInPlace[i, j] = np.sin(2 * np.pi * (1.0 / 100) * i) + np.cos(2 * np.pi * (1.0 / 100) * j)

# Transpose matrixInPlace
matrixInPlace_transposed = np.transpose(matrixInPlace)

# Export original and transposed matrices to CSV, specifying the subfolder
export_matrix_to_csv(matrix1, 'matrix1.csv')
export_matrix_to_csv(matrix2, 'matrix2.csv')
export_matrix_to_csv(checkMatrix, 'checkMatrix.csv')
export_matrix_to_csv(matrix1_transposed, 'matrix1_transposed.csv')
export_matrix_to_csv(matrix2_transposed, 'matrix2_transposed.csv')
export_matrix_to_csv(matrixInPlace, 'matrixInPlace.csv')
export_matrix_to_csv(matrixInPlace_transposed, 'matrixInPlace_transposed.csv')

print(f"CSV Matrices are saved in folder CSV_Dense_Matrices/")
