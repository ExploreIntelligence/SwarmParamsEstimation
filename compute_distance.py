import numpy as np
import pandas as pd

initial_position = np.array(
    [[0.0,0.0],[0.0003, 79.21],[75.33, 24.48],[46.56, -64.08],[-46.56, -64.08],[-75.33, 24.48]])

distance_matrix = np.zeros([6,6])

for row_idx in range(6):
    for col_idx in range(6):
        src = row_idx
        dst = col_idx
        distance_matrix[row_idx][col_idx] = np.linalg.norm(initial_position[src,:]-initial_position[dst,:])

print(distance_matrix)
dis_csv = pd.DataFrame(distance_matrix)
dis_csv.to_csv('results/distance_matrix.csv')