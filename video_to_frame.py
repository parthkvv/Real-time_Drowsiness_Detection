# Program To Read video 
# and Extract Frames 
# import cv2 

# # Function to extract frames 
# def FrameCapture(path): 
	
# 	# Path to video file 
# 	vidObj = cv2.VideoCapture(path) 

# 	# Used as counter variable 
# 	count = 0

# 	# checks whether frames were extracted 
# 	success = 1

# 	while success: 

# 		# vidObj object calls read 
# 		# function extract frames 
# 		success, image = vidObj.read() 

# 		# Saves the frames with frame-count 
# 		cv2.imwrite("frame%d.jpg" % count, image) 

# 		count += 1

# # Driver Code 
# if __name__ == '__main__': 

# 	# Calling the function 
# 	FrameCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4") 


import cv2
import time
from image_detection import detect_faces

while True:
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)
    img_counter = 0
    frame_set = []
    start_time = time.time()

    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= 5: #<---- Check if 5 sec passed
            img_name = "opencv_frame.jpg" #.format(img_counter)
            cv2.imwrite(img_name, frame)
            detect_faces(img_name)
            # print("{} written!".format(img_counter))
            start_time = time.time()
        img_counter += 1

 
        