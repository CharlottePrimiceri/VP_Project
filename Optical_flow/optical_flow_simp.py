import numpy as np
import cv2 as cv
import argparse


path = ("/home/charlotte/Documents/RL-2022/foldprova/segmented_video_car.avi")
cap = cv.VideoCapture(path)
cv.namedWindow('Video', cv.WINDOW_NORMAL)
cv.resizeWindow('Video', 800, 600)
cv.namedWindow('Processed Frame', cv.WINDOW_NORMAL)
cv.resizeWindow('Processed Frame', 800, 600)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

c = 0

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    #path_ov = ("C:/Users/zitoc/opt/output_video")
    path_oi = ("/home/charlotte/Documents/RL-2022/foldprova/out_img")
    image_path = 'path_oi_' + str(c) + '.jpg'
    cv.imwrite(image_path, img)
    c += 1  # Increment the frame number
    #cv.imwrite('path_oi'+str(c) + '.jpg', img)
    #cv.imwrite(path_ov, img)
    cv.imshow('Video', frame)
    cv.imshow('Processed Frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

#cap.release()
#cv.destroyAllWindows()
