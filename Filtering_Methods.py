import numpy as np
from Utils import *


def mean_filter(img,size):
    # Filing with zeros the Kernel matrix
    B = np.zeros((size, size),dtype=np.float32)
    # Applying in every point of matrix the given value 1/size^3
    B.fill(1 / (size) ** 2)
    img_filt = myConv2D(img,B,strides=1,param="same")

    return (img_filt.astype(np.uint8))


"""Traverse the image. For every 3X3 area,
     find the median of the pixels and
     replace the center pixel by the median"""
def median_filter(A):

    m, n = A.shape
    img_filt = np.zeros([m, n])

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = [A[i - 1, j - 1],
                    A[i - 1, j],
                    A[i - 1, j + 1],
                    A[i, j - 1],
                    A[i, j],
                    A[i, j + 1],
                    A[i + 1, j - 1],
                    A[i + 1, j],
                    A[i + 1, j + 1]]

            temp = sorted(temp)
            img_filt[i, j] = temp[4]
    return img_filt.astype(np.uint8)




# Function for deciding the filtering type.
def myImFilter(B,param):
    # Apply Mean if you choose it
    if param =="mean":
        C = mean_filter(B,size=3)
    # Apply Median if you choose it
    elif param == "median":
        C = median_filter(B)
    else:
        print("Invalid param name.!! For filtering plese select mean or median.")
    return C
