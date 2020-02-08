import numpy as np


vector = np.random.randint(1, 20, 15)
print("vector with random values and of size 15 having only Integers in the range 1-20")
print(vector)
# reshape the array to 3 by 5 array = np.reshape(row, column)
print("Reshaping the array to 3 by 5 ")
array = vector.reshape(3, 5)
print(array)

array1 = np.where(array == np.amax(array, axis=1, keepdims=True), 0, array)
print("After replacing  the max in each row by 0")
print(array1)