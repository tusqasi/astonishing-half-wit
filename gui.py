import cv2 as cv
import numpy as np
import matrix_detection
from models.configuration import Config


def main():
    config = Config.load_config()
    print(config)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)
    cv.namedWindow("image")
    cv.createTrackbar("Precision", "image", 0, 100, lambda x: x)
    while True:
        ret, frame = cap.read()
        config.quad_precision = cv.getTrackbarPos("Precision", "image")/10000
        decoded = matrix_detection.decode_matrix(frame, config)
        cv.imshow("thresh", decoded["thres"])
        cv.imshow("image", decoded["final"])
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
