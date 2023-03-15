from math import inf
from typing import List
import numpy as np
import cv2 as cv
import cv2
from pprint import pprint
from itertools import filterfalse as reject
from time import sleep

GREEN = (0, 255, 0)

#    0,    1,  2,      3
# next, prev, fc, parent


def find_inner_contours(heirarchy) -> List[int]:
    return filter(
        lambda x:
        x[1][2] == -1,
        enumerate(heirarchy[0])
    )


def draw_contours_on_image(contours, image):
    if len(contours) == 0:
        return image
    return cv.drawContours(
        image,
        list(zip(*contours))[1],
        -1,
        GREEN,
        thickness=2,
    )


def draw_points_on_image(points, image):
    for x, y in points:
        cv.circle(image, {x, y}, 1, GREEN, thickness=3)
        cv.putText(image, f"{x}, {y}", (x + 10, y), cv, 0.5, GREEN)
    return image


def if_quad(contour):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.1 * peri, True)
    return cv.convexHull(approx, returnPoints=False) == 4,


def find_quads(contours) -> List[int]:
    return list(filter(lambda x: x[1], contours))


def find_inner_most_contours(heirarchy):
    return list(
        map(
            lambda x:
            x[0],
            filter(
                lambda h:
                h[1][2] == -1,
                enumerate(heirarchy)
            )
        )
    )


def find_contour_containg_matrix(heirarchy) -> List[int]:
    return list(
        set(
            map(
                lambda x:
                x[1][3],
                find_inner_contours(heirarchy)
            )
        )
    )


def filter_by_area(contours, min_area: int, max_area: int):
    f = list(
        reject(
            lambda x:
                cv.contourArea(x[1]) > max_area
                or cv.contourArea(x[1]) < min_area,
            contours)
    )
    return f


def decode_matrix(image):

    gray = cv.cvtColor(image,  cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray,  3)

    thresholded = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY, 51, 11)
    contours, heirarchy = cv.findContours(
        blurred,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE)

    contours = list(enumerate(contours))
    contours_with_valid_area = filter_by_area(
        contours,
        10000,
        70000,
    )

    # quads = find_quads(contours)
    # print(f"n_quads {len(quads)}")
    # print(f"n_contours {len(contours)}")
    # print(f"diff {len(contours) - len(quads)}")
    return [draw_contours_on_image(contours, thresholded),]
    # draw_contours_on_image(quads, image)]


def main():
    image = cv.imread("./input_images/3.jpeg")
    decoded_image = decode_matrix(image)
    cv.imshow("frame", decoded_image)
    cv.waitKey()


def with_video():
    cap = cv.VideoCapture(0)
    while True:
        sleep(0.01)
        ret, image = cap.read()
        if ret:
            decoded = decode_matrix(image)
            if isinstance(decoded, list):
                for i, im in enumerate(decoded):
                    cv.imshow(f"frame {i}", im)
        else:
            pass
        cv.imshow("original", image)
        if cv.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    with_video()
