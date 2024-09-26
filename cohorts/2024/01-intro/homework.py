import pandas as pd
import numpy as np
from collections import Counter

"""
1. Select all the "Innjoo" laptops from the dataset.
2. Select only columns `RAM`, `Storage`, `Screen`.
3. Get the underlying NumPy array. Let's call it `X`.
4. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.
5. Compute the inverse of `XTX`.
6. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100]`.
7. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
8. What's the sum of all the elements of the result?
"""

laptops_df = pd.read_csv("laptops.csv")
print(laptops_df.columns)
laptops_df = laptops_df[laptops_df["Brand"] == "Innjoo"]
laptops_df = laptops_df[["RAM", "Storage", "Screen"]]
X = laptops_df.values
XTX = np.dot(X.T, X)
XTX_inverse = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100])
temp = np.dot(XTX_inverse, XTX)
w = np.dot(np.dot(XTX_inverse, X.T), y)
print(np.sum(w))
