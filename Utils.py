import numpy as np

def myConv2D(A, B, strides, param):
    # Defing the output parametres of the Convolution opeeration
    A_pad, pad = pad_Checker(B, A, param)  # We creating the new matrix with padding
    B = np.flipud(np.fliplr(B))
    # Specifing the x,y len of kernel matrix (filter one)
    xKernShape = B.shape[0]
    yKernShape = B.shape[1]

    # Specifing the x,y len of Padded Matrix

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
