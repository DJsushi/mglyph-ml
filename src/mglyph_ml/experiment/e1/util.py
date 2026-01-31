from enum import Enum
from zipfile import ZipFile

import cv2
import numpy as np



class ManifestSampleShape(Enum):
    SQUARE = "s"
    TRIANGLE = "t"
    CIRCLE = "c"


def decode_png_bytes(png_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # HWC, RGB, uint8
    return img


def load_image_into_ndarray(archive: ZipFile, filename: str) -> np.ndarray:
    png_bytes = archive.read(filename)
    return decode_png_bytes(png_bytes)
