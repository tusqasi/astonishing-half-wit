import cv2 as cv
import numpy as np
import matrix_detection
from models.configuration import Config


def nothing(x):
    pass


def main():
    config = Config.load_config()
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)

    # cv.namedWindow("image")
    # cv.createTrackbar('blur_size', 'image', 0, 255, nothing)
    # cv.createTrackbar('area_max', 'image', 100_000, 100_000_000, nothing)
    # cv.createTrackbar('area_min', 'image', 100, 100_000, nothing)
    while True:
        ret, frame = cap.read()
        # config.area_max = cv.getTrackbarPos('area_max', "image")
        # config.area_min = cv.getTrackbarPos('area_min', "image")

        decoded = matrix_detection.decode_matrix(frame, config)
        cv.imshow("image", decoded["final"])
        cv.imshow("thresh", decoded["thres"])
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
