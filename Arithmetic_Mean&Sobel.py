import numpy as np
import cv2
import skvideo
skvideo.setFFmpegPath("C:/ffmpeg/bin/")
import skvideo.io


def ArithmeticMean(A, param):

    A_pad, pad = pad_Checker(A, param)
    output = np.zeros(A_pad.shape)

    for z in range(1, A_pad.shape[0]):
        for i in range(1, A_pad.shape[1]):
            for j in range(1, A_pad.shape[2]):
                output[z,i,j] = np.sum(A_pad[z-1:z+1,i-1:i+1,j-1:j+1])
    output = output * (1 / 27)
    output = output.astype(np.uint8)

    return output


def mySobel(A, param, strides):

    # check for param and add or no zero-padding
    A_pad, pad = pad_Checker( A, param)

    # create the kernels for sobel edge detection in horizontal and vertical dim
    Gx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # fliping the kernels
    Gx = np.flipud(np.fliplr(Gx))
    Gy = np.flipud(np.fliplr(Gy))

    # extract shape of the video
    xApadShape = A_pad.shape[2]
    yApadShape = A_pad.shape[1]
    zApadShape = A_pad.shape[0]

    #Definig the output matrix shape.
    xOutput = int(((xApadShape - 3 + 2 * pad) / strides) + 1)
    yOutput = int(((yApadShape - 3 + 2 * pad) / strides) + 1)
    zOutput = int(((zApadShape - 3 + 2 * pad) / strides) + 1)
    output = np.zeros((zOutput, yOutput, xOutput))  # Filling with zeros our new output matrix

        #Starting the convolution operator

    for z in range(A_pad.shape[0]):                      #taking notice of Depth column
        if z > A_pad.shape[0]-3:
            break

        for y in range(A_pad.shape[1]):

                if y > A_pad.shape[1] - 3:     # Go to next row once kernel is out of bounds
                    break

                for x in range(A_pad.shape[2]):

                        if x > A_pad.shape[2] - 3:
                            break
                        try:

                            # Making the Convolution operator. Passing first horizontal and then vertical
                            gx = np.sum(np.multiply(Gx, A_pad[z, y:y + 3, x:x + 3]))
                            gy = np.sum(np.multiply(Gy, A_pad[z, y:y + 3, x:x + 3]))
                            # concatenate in the same output
                            output[z, y + 1, x + 1] = np.sqrt( gy ** 2 + gx ** 2)
                        except:
                                break
    return output






# With this def i am applying zero padding if the param ="same" and unpadding the outpout video of covnvolution when param="unpad"
# and when we insert "valid" in param we dont apply zero padding in video
def pad_Checker( A, param):
    n = 3
    padv = int((n - 1) / 2)           # so the equation out=(input-kernel+2*padding)/strides+1 is converted to p=(kernel-1)/2 when
    # When we defing that param='same' we defing that our output shape should be equal with input shape
    if param == 'same':

        C = pad_video(A, size=padv)
    # When we defing that param='valid' we don not apply padding in video
    elif param == "valid":
        padv = 0
        C = A

    # In any other word than "same" and "valid" we rise error message
    else:
        print("No such value available.! Please select same or valid.!")
    return C, padv

def pad_video(A, size):

    #Apply equal padding to all sides
    A_pad = np.zeros((A.shape[0] + 2*size, A.shape[1] + 2*size, A.shape[2] + 2*size))
    A_pad[1*size:-1*size, 1*size:-1*size, 1*size:-1*size] = A   #applying in new matrix the original video matrix leaving a outer shell full of zeros

    return A_pad

def show_vid(name, video):

    cap = cv2.VideoCapture(video)

    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            # we show each frame of the video
            cv2.imshow(name,frame)
            if cv2.waitKey(25) & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    vid = skvideo.io.vread("resources/video.mp4")          #Reading the video from disc
    A_or = np.empty_like(vid[..., 0])    #Creating the video matrix A_or without values in it


    # Keeping the matrix A without the rgb channels (making it in gray scale)
    for i in range(vid.shape[0]):
        A_or[i] = cv2.cvtColor(vid[i], cv2.COLOR_RGB2GRAY)

    # Applying the Arithemtic Mean in the Grayscale video.
    Arith_Mean = ArithmeticMean(A_or, param="same")

    # Applying Sobel Edge Detection on Grayscale video.
    Sobel = mySobel(A_or, param="same", strides=1)


    #Starting the writting operation of the Arithmetic Mean.
    writer = skvideo.io.FFmpegWriter("output_1.avi")
    for i in range(Arith_Mean.shape[0]):
        writer.writeFrame(Arith_Mean[i, :, :])
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    writer.close()

    #display our video for Arithmetic Mean
    show_vid(name = "Arithmetic_Mean", video="output_1.avi")

    # Lets start write the Sobel video

    writer=skvideo.io.FFmpegWriter("output_2.avi")
    for i in range(Sobel.shape[0]):
        writer.writeFrame(Sobel[i, :, :])
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    writer.close()

    # display our video for Sobel edge Detection
    show_vid(name = "Sobel_Edge_Detection",video="output_2.avi")

if __name__ == "__main__":
    main()
