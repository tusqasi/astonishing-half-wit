import cv2 
import functools

url =  "http://192.168.126.238:8080/video"
# open the feed
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(url)

while True:
    # read next frame
     ret, frame = cap.read()
    
     # cv2.imshow('frame', frame)

     # continue
     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
     gausBlur = cv2.GaussianBlur(gray, (9,9),0)
     #cv2.imshow('frame', gray)

     cv2.imshow('blurred', gausBlur)
     th3 = cv2.adaptiveThreshold(gausBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,2)
     cv2.imshow('thresholded', th3)
     # contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     # print(contours)

     # cv2.drawContours(frame, contours, -1, (0,255,0), 10)

     # cv2.imshow('With countour', frame) 
    
    # if user presses q quit program
     if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# close the connection and close all windows
cap.release()
cv2.destroyAllWindows()
