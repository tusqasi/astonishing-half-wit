import cv2 

# insert the HTTP(S)/RSTP feed from the camera
url = "http://192.168.126.238:8080/video"

# open the feed
cap = cv2.VideoCapture(url)

while True:
    # read next frame
     ret, frame = cap.read()
    
    # show frame to user
     cv2.imshow('frame', frame)
    
    # if user presses q quit program
     if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# close the connection and close all windows
cap.release()
cv2.destroyAllWindows()
