#import necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as numpy
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    #compute euclidean distance between the eyes
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
agrs = vars(ap.parse_args)

#define two constants, onr for the eye aspect ration to indicate blink
#second constant for the number of consecutive frames the eye must be below threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

#inititalize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

#inititalize the dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#grabing the indexes of facial lanmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#starting the video feed
#vs = FileVideoStream(args["video"]).start()
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

#loop over the frames from the video stream
while True:
    #check if its file video stream, we also need to check if the more frames are left in the buffer
    if fileStream and not vs.more():
        break
    
    #grabing the frame from the video
    frame = vs.read()
    #resize the frame to width of 450
    frame = imutils.resize(frame, width = 450)
    #convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting face in the frame
    rects = detector(gray, 0)

    #loop over the face detections
    for rect in rects:
        #detecting facial landmarks for face rigion, convert the facial landmark to NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #extracting the left and right eye
        leftEye = shape[lStart: lEnd]
        rightEye = shape[rStart: rEnd]
        
        #calculating EAR for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #average the eye aspect ratio for both eyes
        ear = (leftEAR+rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #check to see if the eye aspect ratio is below the blink threshold, increament the count if so
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            #check if eye is closed for fixed number of frames, if closed for 3 frames(in this case), increase totalcount and reset count to 0
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

            #draw the total number of blinks on the  frame along with the computed ear for the frame
            cv2.putText(frame, "Blinks{}".format(TOTAL),(10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear),(300, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        #show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #if 'q' was pressed, break the loop
        if key == ord("q"):
            break

#cleanup
cv2.distroyAllWindows()
vs.stop()