from typing import Tuple

import cv2
import numpy as np
import pytesseract


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):
    return cv2.medianBlur(image, 3)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def recognize_image(image: np.ndarray, langs: Tuple[str, ...] = ("eng", "rus")) -> str:
    image = get_grayscale(image)
    image = thresholding(image)
    image = remove_noise(image)
    image = deskew(image)

    langs = "+".join(langs)
    decoded_text = pytesseract.image_to_string(image, lang=langs)

    return decoded_text


def recognize_binary_image(
    image: bytes, langs: Tuple[str, ...] = ("eng", "rus")
) -> str:
    image = np.frombuffer(image, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)

    return recognize_image(image)