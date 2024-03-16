import numpy as np

def divide_2d_array(array):
    if array.shape[0] % 2 != 0 or array.shape[1] % 2 != 0:
        raise ValueError("Array dimensions must be even to divide into 4 equal parts.")

    half_rows = array.shape[0] // 2
    half_cols = array.shape[1] // 2

    first_quarter = array[:half_rows, :half_cols]
    second_quarter = array[:half_rows, half_cols:]
    third_quarter = array[half_rows:, :half_cols]
    fourth_quarter = array[half_rows:, half_cols:]

    return first_quarter, second_quarter, third_quarter, fourth_quarter

# Example usage:
array = np.array([[1, 2, 3, 4,5],
                  [5, 6, 7, 8,5],
                  [9, 10, 11, 12,5],
                  [13, 14, 15, 16,5]])

result = divide_2d_array(array)
print("First quarter:\n", result[0])
print("Second quarter:\n", result[1])
print("Third quarter:\n", result[2])
print("Fourth quarter:\n", result[3])
