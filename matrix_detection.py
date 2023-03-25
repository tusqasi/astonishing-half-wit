from math import inf
from typing import List, Dict
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
    return [x for x in enumerate(heirarchy[0])
            if x[1][2] == -1]


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


def is_quad(contour, precision):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, precision * peri, True)
    size_contours = len(contour)
    print(size_contours)
    size_approx = len(approx)
    return  6 > size_approx > 3


def find_quads(contours, precision: float) -> Dict:
    return {k: v for k, v in contours.items() if is_quad(v, precision)}


def find_inner_most_contours(heirarchy):
    return [h[1] for h in heirarchy if h[1][2] == -1]


def find_contour_containg_matrix(heirarchy) -> List[int]:
    inner_contours = find_inner_contours(heirarchy)
    x = Counter(x[1][3] for x in inner_contours)
    possible = []
    for parent, count in x.items():
        if count <= 4:
            possible.append(parent)
    return possible


def filter_by_area(contours, min_area: int, max_area: int):
    ret_val = {}
    for k, v in contours.items():
        area = cv.contourArea(v)
        if area < max_area and area > min_area:
            ret_val[k] = v
    return ret_val


def find_extreme_points(contour, precision):
    peri = cv.arcLength(contour, True)
    return cv.approxPolyDP(contour, precision * peri, True)


def find_external_point(image):
    contours = cv2.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def find_quads_2(contours):
    ret_val = {}
    for k, c in contours:
        rot_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)
        ret_val[k] = box
    return ret_val


def decode_matrix(image, config: Config):
    image = cv2.fastNlMeansDenoising(image,  None, 4, 3, 5)
    gray = cv.cvtColor(image,  cv.COLOR_BGR2GRAY)
    # blurred = cv2.medianBlur(gray, 5)
    blurred = cv2.GaussianBlur(gray, (3, 3), 3, 3)
    blurred = gray
    if config.adaptive_threshold:
        morph = cv.adaptiveThreshold(
            blurred, config.threshold_max, cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY_INV, config.block_size,  config.c)
    else:
        _, morph = cv.threshold(
            blurred,
            config.threshold,
            config.threshold_max,
            cv.THRESH_BINARY_INV
        )

    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    kernel_dialate = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel_close)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel_open)
    morph = cv.dilate(morph, kernel_open, iterations=1)

    contours, heirarchy = cv.findContours(
        morph,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE,
    )
    # contours, heirarchy = cv.findContours(
    #     morph,
    #     cv.RETR_EXTERNAL,
    #     cv.CHAIN_APPROX_SIMPLE,
    # )
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

    ret_val = {
        "thres": morph
    }
    ret_val["final"] = cv.cvtColor(
        draw_contours_on_image(
            contours_with_quads,
            cv2.cvtColor(blurred, cv.COLOR_BGR2RGB),
        ),
        cv.COLOR_BGR2RGB,
    )
    return ret_val
