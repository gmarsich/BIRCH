import numpy as np
import math

list = np.array([1, 2, 3])
list_2 = np.array([1, 1, 1])



pow_2 = np.sum((list - list_2) ** 2)
print(pow_2 ** (1/2))
    # return pow_2 ** (1/2)