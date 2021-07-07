import numpy as np
import cv2
import skvideo
import skvideo.io


def myConv3D(A,B,strides,param):
    output= None                              #Defing the output parametre of the Convolution opeeration
    A_pad,pad=pad_Checker(B, A,output,param)  # We creating the new matrix with padding

    # Specifing the x,y,z len of kernel matrix (filter one)

    xKernShape = B.shape[2]
    yKernShape = B.shape[1]
    zKernShape = B.shape[0]

    # Specifing the x,y,z len of Padded Matrix

    xApadShape = A_pad.shape[2]
    yApadShape = A_pad.shape[1]
    zApadShape = A_pad.shape[0]

    #Definig the output matrix shape.

    xOutput = int(((xApadShape - xKernShape + 2 * pad) / strides) + 1)
    yOutput = int(((yApadShape - yKernShape + 2 * pad) / strides) + 1)
    zOutput = int(((zApadShape - zKernShape + 2 * pad) / strides) + 1)
    output = np.zeros((zOutput,yOutput,xOutput))                           # Filling with zeros our new output matrix

    #Starting the convolution operator

    for z in range(A_pad.shape[0]):                      #taking notice of Depth column
        if z > A_pad.shape[0]-zKernShape:
            break

        for y in range(A_pad.shape[1]):

                if y > A_pad.shape[1] - yKernShape:     # Go to next row once kernel is out of bounds
                    break

                for x in range(A_pad.shape[2]):

                        if x > A_pad.shape[2] - xKernShape:
                            break
                        try:
                            # Making the Convolution operator.

                            output[z,y,x] = (B * A_pad[ z:z + zKernShape, y: y + yKernShape,x: x + xKernShape]).sum()

                        except:
                               break

    #Unpadding the output video. To achive that we define param='unpad'
    param='unpad'
    output,pad=pad_Checker(B,A,output,param)

    return output

def create_smooth_kernel(size):

    Xsize = size
    Ysize = size
    Zsize = size
    B = np.zeros((Xsize,Ysize,Zsize))         #Filing with zeros the Kernel matrix
    B.fill(1/(size)**3)                       #Applying in every point of matrix the given value 1/size^3

    return B


# With this def i am applying zero padding if the param ="same" and unpadding the outpout video of covnvolution when param="unpad"
# and when we insert "valid" in param we dont apply zero padding in video
def pad_Checker(B, A, A_out, param):
    n, n, n = B.shape
    padv = int((n - 1) / 2)           # so the equation out=(input-kernel+2*padding)/strides+1 is converted to p=(kernel-1)/2 when

    if param == 'same':               # When we defing that param='same' we defing that our output shape should be equal with input shape

        C = pad_image(A, size=padv)
    elif param == 'unpad':

        C = unpad(A, A_out, size=padv)
    else:
        padv = 0
        C = A
    return C, padv

# With the follow def we unpadding the output matrix
def unpad(A, A_out, size):
    A_unpad = np.zeros((A.shape[0], A.shape[1], A.shape[2]))
    A_unpad[:, :, :] = A_out[size:-size, size:-size, size:-size]
    return A_unpad

def pad_image(A, size):

    #Apply equal padding to all sides
    A_pad = np.zeros((A.shape[0] + 2*size, A.shape[1] + 2*size, A.shape[2] + 2*size))
    A_pad[1*size:-1*size, 1*size:-1*size, 1*size:-1*size] = A   #applying in new matrix the original video matrix leaving a outer shell full of zeros

    return A_pad



def main():
    vid = skvideo.io.vread("resources/video.mp4")          #Reading the video from disc
    A_or = np.empty_like(vid[..., 0])    #Creating the video matrix A_or without values in it
    print(A_or.shape)


    # Keeping the matrix A without the rgb channels (making it in gray scale)
    for i in range(vid.shape[0]):
        A_or[i] = cv2.cvtColor(vid[i], cv2.COLOR_RGB2GRAY)




    strides=1             # We define the step of pixels shifts over the input matrix. is 1 we move the filters to 1 pixel at a time
    size=3                # Defining  the size of the filter (Kernel) and strides.
                          ## strides(the step of our moving filter) is setted to be 1.
    param = "same"        # We setting the param='same' since we want our output video have the same size with input and because that
                          # we will apply zero padding on it

    # Creating the Kernel.
    K = create_smooth_kernel(size)

    # Start the convolution operation with param='same' which defines that we will apply zero padding in our video
    #to make sure that our output will have the same shape as our input video.

    Conv = myConv3D(A_or, K,strides,param)


    #Starting the writting operation of our new Convolvied video.

    writer=skvideo.io.FFmpegWriter("Convolution.mp4")
    for i in range(Conv.shape[0]):

        writer.writeFrame(Conv[i, :, :])

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    writer.close()

if __name__ == "__main__":
    main()
