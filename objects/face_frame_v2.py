"""
PILに対応
フレーム表示を円形に
"""



# ライブラリインポート
from multiprocessing import Process, Manager, Value
import pickle
import math
import time
import sys
import traceback
import random

from PIL import Image, ImageDraw
import cv2
import face_recognition
import dlib
import numpy as np


class FaceFrame:
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def cal_center(self):
        y = self.top + (self.bottom - self.top) / 2
        x = self.left + (self.right - self.left) / 2
        center = [y, x]
        return center

    def draw_frame(self, frame):
        draw = ImageDraw.Draw(frame)
        draw.ellipse((self.left, self.top, self.right, self.bottom), outline=(255, 0, 0))
        return frame

    def setter(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
