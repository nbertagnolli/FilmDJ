__author__ = 'Nicolas Bertagnolli'

import cv2
import numpy as np
import math


def nothing(x):
    pass


def main():
    # Load in videos
    movie1 = cv2.VideoCapture('../../../../video/anni003.mpg')
    movie2 = cv2.VideoCapture('../../../../video/anni007.mpg')

    # Get video size
    # TODO:: CHeck to make sure both videos are the same size
    ret1, frame1 = movie1.read()
    rows, cols, dim = frame1.shape

    # Create dummy image to hold video edits
    display = np.zeros((rows, 2 * cols, dim), np.uint8)
    mixed = np.zeros((rows, cols, dim), np.uint8)

    # Create window
    cv2.namedWindow('video')
    cv2.namedWindow('mixed')

    # Add a slider
    cv2.createTrackbar('alpha', 'video', 0, 100, nothing)

    # create trackbars for color change
    cv2.createTrackbar('R', 'video', 0, 100, nothing)
    cv2.createTrackbar('G', 'video', 0, 100, nothing)
    cv2.createTrackbar('B', 'video', 0, 100, nothing)

    while movie1.isOpened() and movie2.isOpened():
        # Define interrupt key as 'esc'
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current frame of video
        ret1, frame1 = movie1.read()
        ret2, frame2 = movie2.read()

        # Split frame into rgb channels
        b1, g1, r1 = cv2.split(frame1)
        b2, g2, r2 = cv2.split(frame2)

        # get current positions of trackbars
        r_percent = cv2.getTrackbarPos('R', 'video') / 100.0
        g_percent = cv2.getTrackbarPos('G', 'video') / 100.0
        b_percent = cv2.getTrackbarPos('B', 'video') / 100.0
        alpha = cv2.getTrackbarPos('alpha', 'video') / 100.0

        # Create combined display for first movie
        display[:, :cols, 0] = b1 * b_percent
        display[:, :cols, 1] = g1 * g_percent
        display[:, :cols, 2] = r1 * r_percent
        # Create combined display for second movie
        display[:, cols:, 0] = b2 * b_percent
        display[:, cols:, 1] = g2 * g_percent
        display[:, cols:, 2] = r2 * r_percent

        # Create mixed channel
        mixed[:, :, 0] = display[:, :cols, 0] * alpha + display[:, cols:, 0] * (1 - alpha)
        mixed[:, :, 1] = display[:, :cols, 1] * alpha + display[:, cols:, 1] * (1 - alpha)
        mixed[:, :, 2] = display[:, :cols, 2] * alpha + display[:, cols:, 2] * (1 - alpha)


        # Show video in window
        cv2.imshow("video", display)
        cv2.imshow('mixed', mixed)


    # When escape is pressed destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


