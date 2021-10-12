import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture("E:\\StudyData\\GroundTruthData\\PIS-2212\\Resting2.mp4") # change video and participant id here
fps = round(cam.get(cv2.CAP_PROP_FPS))
c=1
T = 1/fps
try:
    pathval = "E:\\StudyData\\GroundTruthData\\PIS-2212\\Resting2"
    # creating a folder named data
    if not os.path.exists(pathval):
        os.makedirs(pathval)

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while (True):

    # reading from frame
    ret, frame = cam.read()

    if ret:
        if (c % fps == 0):
            # if video is still left continue creating images
            name = pathval + '\\frame' + str(c) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        c += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()