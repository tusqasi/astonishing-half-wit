import cv2 as cv
import numpy as np
import matrix_detection
from models.configuration import Config


def main(retries):
    config = Config.load_config()
    cap = cv.VideoCapture(0)
    if not cap.grab():
        print("Video Not working")
        print("Retrying")
        return main(retries -1)
    elif not retries:
        print("Done retrying")
        return None

    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)

    while True:
        ret, frame = cap.read()

        decoded = matrix_detection.decode_matrix(frame, config)
        cv.imshow("image", decoded["final"])
        cv.imshow("thresh", decoded["thres"])
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main(3)
