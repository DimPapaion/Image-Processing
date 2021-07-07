import numpy as np
import cv2
import random

def myConv2D(A, B, strides, param):
    output = None                                 # Defing the output parametre of the Convolution opeeration
    A_pad, pad = pad_Checker(B, A, param)  # We creating the new matrix with padding
    B = np.flipud(np.fliplr(B))
    # Specifing the x,y len of kernel matrix (filter one)
    xKernShape = B.shape[0]
    yKernShape = B.shape[1]

    # Specifing the x,y,z len of Padded Matrix

    xApadShape = A_pad.shape[0]
    yApadShape = A_pad.shape[1]

    # Definig the output matrix shape.

    xOutput = int(((xApadShape - xKernShape + 2 * pad) / strides) + 1)
    yOutput = int(((yApadShape - yKernShape + 2 * pad) / strides) + 1)

    output = np.zeros((xOutput, yOutput),dtype=np.float32)  # Filling with zeros our new output matrix

    # Starting the convolution operator

    for y in range(A_pad.shape[1]):

        if y > A_pad.shape[1] - yKernShape:  # Go to next row once kernel is out of bounds
            break

        if y % strides == 0:

            for x in range(A_pad.shape[0]):

                if x > A_pad.shape[0] - xKernShape:
                    break
                try:
                    if x % strides ==0:
                    # Making the Convolution operator.

                        output[x, y] = (B * A_pad[x: x + xKernShape, y: y + yKernShape]).sum()

                except:
                    break
    return output



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

def mean_filter(img,size):

    B = np.zeros((size, size),dtype=np.float32)  # Filing with zeros the Kernel matrix
    B.fill(1 / (size) ** 2)  # Applying in every point of matrix the given value 1/size^3

    img_filt = myConv2D(img,B,strides=1,param="same")

    return (img_filt.astype(np.uint8))

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

def median_filter(A):
    m, n = A.shape

    # Traverse the image. For every 3X3 area,
    # find the median of the pixels and
    # replace the ceter pixel by the median
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

# With this def i am applying zero padding if the param ="same" and unpadding the outpout video of covnvolution when param="unpad"
# and when we insert "valid" in param we dont apply zero padding in video
def pad_Checker(B, A, param):
    n, n = B.shape
    padv = int((n - 1) / 2)  # so the equation out=(input-kernel+2*padding)/strides+1 is converted to p=(kernel-1)/2 when

    if param == 'same':  # When we defing that param='same' we defing that our output shape should be equal with input shape
        C = pad_image(A, size=padv)

    elif param == 'valid':
        padv = 0
        C = A
    else:
        print("Invalid param name.!! For noising please select same or valid.")
    return C, padv


def pad_image(A, size):
    # Apply equal padding to all sides
    A_pad = np.zeros((A.shape[0] + 2 * size, A.shape[1] + 2 * size), dtype=np.float32)
    A_pad[1 * size:-1 * size,1 * size:-1 * size] = A  # applying in new matrix the original video matrix leaving a outer shell full of zeros
    return A_pad

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

# Function for deciding the filtering type.
def myImFilter(B,param):
    # Apply mean if you chose it
    if param =="mean":
        C = mean_filter(B,size=3)
    # Apply media if you chose it
    elif param == "median":
        C = median_filter(B)
    else:

        print("Invalid param name.!! For filtering plese select mean or median.")
    return C

def main():
    # ============================================#
    #        Reading the image from disc          #
    # ============================================#

    image = cv2.imread("house.png")
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)


    # ===========================================#
    #   Converting image to GrayScale            #
    # ===========================================#

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Gray scale image: ", imgGray.shape,len(imgGray.shape))
    cv2.imshow("Gray Image", imgGray)
    cv2.waitKey(0)
    cv2.imwrite("Gray_Scale_Image.png", imgGray)

    # ==========================================#
    # Adding Salt and Pepper noise to image     #
    # ==========================================#

    SPnoise_img = myImNoise(imgGray, param="saltandpepper")
    cv2.imshow('Salt And Pepper Noise Image', SPnoise_img)
    cv2.waitKey(0)
    cv2.imwrite('Salt_And_Pepper_Noise_Image.png', SPnoise_img)

    # =======================================#
    #      Adding noise Gaussian to image    #
    # =======================================#

    Gnoise_img = myImNoise(imgGray, param="gaussian")
    cv2.imshow('Gaussian Noise Image ', Gnoise_img)
    cv2.waitKey(0)
    cv2.imwrite('Gaussian_Noise_Image.png', Gnoise_img)

    # =======================================#
    #      Mean filter Gaussian noise image  #
    # =======================================#

    Mean = myImFilter(Gnoise_img, param="mean")
    cv2.imshow('Mean filtered Image ', Mean)
    cv2.waitKey(0)
    cv2.imwrite('Mean_filtered_Image.png', Mean)

    # =======================================#
    #   Median filter S&P noise image        #
    # =======================================#
    Median = myImFilter(SPnoise_img, param="median")
    cv2.imshow('Median filtering Image ', Median)
    cv2.waitKey(0)
    cv2.imwrite('Median_filtering_Image.png', Median)

if __name__ == "__main__":
    main()
