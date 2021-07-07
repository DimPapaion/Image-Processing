import numpy as np
import cv2
import random
from Utils import *
from Noise_Filters import *
from Filtering_Methods import *

def main():
    # ============================================#
    #        Reading the image from disc          #
    # ============================================#

    image = cv2.imread("resources/house.jpg")
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)


    # ===========================================#
    #   Converting image to GrayScale            #
    # ===========================================#

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Gray scale image: ", imgGray.shape,len(imgGray.shape))
    cv2.imshow("Gray Image", imgGray)
    cv2.waitKey(0)
    cv2.imwrite("Results/Gray_Scale_Image.png", imgGray)

    # ==========================================#
    # Adding Salt and Pepper noise to image     #
    # ==========================================#

    SPnoise_img = myImNoise(imgGray, param="saltandpepper")
    cv2.imshow('Salt And Pepper Noise Image', SPnoise_img)
    cv2.waitKey(0)
    cv2.imwrite('Results/Salt_And_Pepper_Noise_Image.png', SPnoise_img)

    # =======================================#
    #      Adding noise Gaussian to image    #
    # =======================================#

    Gnoise_img = myImNoise(imgGray, param="gaussian")
    cv2.imshow('Gaussian Noise Image ', Gnoise_img)
    cv2.waitKey(0)
    cv2.imwrite('Results/Gaussian_Noise_Image.png', Gnoise_img)

    # =======================================#
    #      Mean filter Gaussian noise image  #
    # =======================================#

    Mean = myImFilter(Gnoise_img, param="mean")
    cv2.imshow('Mean filtered Image ', Mean)
    cv2.waitKey(0)
    cv2.imwrite('Results/Mean_filtered_Image.png', Mean)

    # =======================================#
    #   Median filter S&P noise image        #
    # =======================================#
    Median = myImFilter(SPnoise_img, param="median")
    cv2.imshow('Median filtering Image ', Median)
    cv2.waitKey(0)
    cv2.imwrite('Results/Median_filtering_Image.png', Median)

if __name__ == "__main__":
    main()
