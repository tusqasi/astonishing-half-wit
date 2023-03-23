from math import inf
from typing import List
import numpy as np
import cv2 as cv
import cv2
from pprint import pprint
from itertools import filterfalse as reject
from time import sleep
from models.configuration import Config
from collections import Counter

GREEN = (0, 255, 0)

#    0,    1,  2,      3
# next, prev, fc, parent


def find_inner_contours(heirarchy) -> List[int]:
    return list(filter(
        lambda x:
        x[1][2] == -1,
        enumerate(heirarchy[0])
    ))


def draw_contours_on_image(contours: dict, image):
    if len(contours.values()) == 0:
        return image
    return cv.drawContours(
        image,
        list(contours.values()),
        -1,
        GREEN,
        thickness=2,
    )


def draw_points_on_image(points, image):
    for x, y in points:
        cv.circle(image, {x, y}, 1, GREEN, thickness=3)
        cv.putText(image, f"{x}, {y}", (x + 10, y), cv, 0.5, GREEN)
    return image


def if_quad(contour, precision):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, precision * peri, True)
    return cv.convexHull(approx, returnPoints=False) == 4,


def find_quads(contours, precision) -> List[int]:
    return {k: v for k, v in contours.items() if if_quad(v, precision)}


def find_inner_most_contours(heirarchy):
    return list(
        map(
            lambda k, v:
            v,
            filter(
                lambda h:
                h[1][2] == -1,
                enumerate(heirarchy)
            )
        )
    )


def find_contour_containg_matrix(heirarchy) -> List[int]:
    inner_contours = find_inner_contours(heirarchy)
    x = Counter(x[1][3] for x in inner_contours)
    possible = []
    for parent, count in x.items():
        if count <= 4:
            possible.append(parent)
    return possible


def filter_by_area(contours, min_area: int, max_area: int):
    return {
        k: v for k, v in contours.items()
        if cv.contourArea(v) > max_area
        or cv.contourArea(v) < min_area
    }


def find_extreme_points(contour, precision):
    peri = cv.arcLength(contour, True)
    return cv.approxPolyDP(contour, precision * peri, True)


def decode_matrix(image, config: Config):
    gray = cv.cvtColor(image,  cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray,  config.blur_size)
    if config.adaptive_threshold:
        thresholded = cv.adaptiveThreshold(
            blurred, config.threshold_max, cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY_INV, config.block_size,  config.c)
    else:
        _, thresholded = cv.threshold(
            blurred,
            config.threshold,
            config.threshold_max,
            cv.THRESH_BINARY_INV
        )

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    morph = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel_close)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel_open)
    contours, heirarchy = cv.findContours(
        morph,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE,
    )

    contours = dict(enumerate(contours))

    contours_with_valid_area = filter_by_area(
        contours,
        config.area_min,
        config.area_max,
    )
    contours_with_quads = find_quads(
        contours_with_valid_area,
        config.quad_precision,
    )
    contours_with_matrix = find_contour_containg_matrix(heirarchy)

    ret_val = {
        "thres": morph
    }
    with_quads_and_area = {
        k: v for k, v in contours_with_quads.items()
        if k in contours_with_matrix
    }
    ret_val["final"] = cv.cvtColor(
        draw_contours_on_image(
            with_quads_and_area,
            image,
        ),
        cv.COLOR_BGR2RGB,
    )
    return ret_val
