import numpy as np
import cv2

import skvideo
skvideo.setFFmpegPath("C:/ffmpeg/bin/")
import matplotlib.pyplot as plt
import skvideo.io

vid = skvideo.io.vread("resources/video.mp4")
print(vid.shape)

vid_r, vid_g, vid_b = vid[:,:,:,0], vid[:,:,:,1], vid[:,:,:,2]
print(vid_r.shape, vid_b.shape, vid_g.shape)
A_pad = np.empty_like(vid[..., 0])  # Creating the video matrix A_or without values in it
print(A_pad.shape)
for i in range(vid.shape[0]):
    A_pad[i] = cv2.cvtColor(vid[i], cv2.COLOR_RGB2GRAY)
print(A_pad.shape)
output = np.zeros(A_pad.shape)

for z in range(1,A_pad.shape[0]):
    for i in range(1,A_pad.shape[1]):
        for j in range(1, A_pad.shape[2]):
            output[z,i,j] = np.sum(A_pad[z-1:z+1,i-1:i+1,j-1:j+1])
output = output *(1/27)
output = output.astype(np.uint8)
# Keeping the matrix A without the rgb channels (making it in gray scale)


writer=skvideo.io.FFmpegWriter("ArithmeticMean.mp4")
for i in range(output.shape[0]):

    writer.writeFrame(output[i, :, :])

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

