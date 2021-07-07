import numpy as np
from Utils import *
import random

def GaussianBlurImage( A,sigma):
    image = np.array(A, dtype=np.float32)
    filter_size =  2*int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m,m+1):
        for y in range(-n,n+1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x +m, y+n] = (1 / x1) * x2

    img_noise= myConv2D(image, gaussian_filter, strides=1, param="same")
    return (img_noise.astype(np.uint8))


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob                    # threshold for our probabiblity of appearing noise
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# Def for deciding which noise should we implement.
def myImNoise(A,param):
    # Apply salt and peper if we chose it
    if param =="saltandpepper":
        B = sp_noise(A, prob=0.05)

    #Apply gaussian if we chose it
    elif param == "gaussian":
        B = GaussianBlurImage(A, sigma=1)

    else:

        print("Invalid param name.!! For noising please select saltandpepper or gaussian.")
    return B