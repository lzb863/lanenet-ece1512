import cv2 as cv
from detector import do_canny, do_segment, calculate_lines, visualize_lines, config
import matplotlib.pyplot as plt
import numpy as np
import os
# The video feed is read in as a VideoCapture object


image_folder = '/Users/sajadnorouzi/Documents/Winter2019/ECE1512/Assignment3/lane-detector/images'

#
# cap = cv.VideoCapture("input.mp4")
# while (cap.isOpened()):
#     # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
#     ret, frame = cap.read()

for index, filename in enumerate(os.listdir('./images')):
    frame = np.array(cv.imread(image_folder+'/'+filename))
    #ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    print('frame shape: ', index, frame.shape)

    print(filename)
    frame = frame[:-100, :, :]
    # print(filename)
    print('frame shape: ', index, frame.shape)

    canny = do_canny(frame)
    height = canny.shape[1]
    average = np.nonzero(canny[:, int(height*0.7):])

    average_x  = np.mean(average[0])
    print(average_x)
    plt.imshow(canny)
    #plt.show()
    segment = do_segment(canny, average_x)
    plt.imshow(segment)
    #plt.show()
    hough = cv.HoughLinesP(segment, rho=1, theta=np.pi/180, threshold=config['threshold'],
                           minLineLength=config['minLineLength'], maxLineGap=config['maxLineGap'])

    #hough = cv.HoughLinesP(segment, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)

    lines = calculate_lines(frame, hough)
    # Visualizes the lines
    lines_visualize = visualize_lines(frame, lines)
    # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
    output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
    # Opens a new window and displays the output frame
    cv.imshow("output", output)
    input()




# The following frees up resources and closes all windows
#cap.release()
# cv.destroyAllWindows()
