"""
移動量の更新（定数）をなくした
"""


from PIL import Image, ImageDraw, ImageFont
import math


class Line:
    def __init__(self, x0=0, y0=0, x1=0, y1=0, movement_amount_x=0, movement_amount_y=0):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.cross_x = 0
        self.cross_y = 0
        self.movement_amount_x = movement_amount_x
        self.movement_amount_y = movement_amount_y

    def draw_line(self, frame, rect):
        """
        rect = (left, top, right, bottom)
        """
        left, top, right, bottom = rect
        circle_center_x = left + (right-left) / 2
        circle_center_y = top + (bottom-top) / 2
        radius = (right-left) / 2
        destination_x = circle_center_x + self.movement_amount_x
        destination_y = circle_center_y + self.movement_amount_y
        theta = math.atan2(destination_y-circle_center_y, destination_x-circle_center_x)
        self.cross_x = radius*math.cos(theta)+circle_center_x
        self.cross_y = radius*math.sin(theta)+circle_center_y

        if ((self.x1-circle_center_x)**2+(self.y1-circle_center_y)**2)<radius**2:
            print("領域外のため描画せず.パラメータ:{}".format((self.x1-circle_center_x)**2+(self.y1-circle_center_y)))
            return

        draw = ImageDraw.Draw(frame)
        draw.line((self.cross_x, self.cross_y, self.x1, self.y1), fill=(255, 255, 255), width=1)

        print("描いた線の座標位置:{}, {}, {}, {}".format(self.x0, self.y0, self.x1, self.y1))

    def setter(self, x0, y0, x1, y1):
        self.x0 = int(x0)
        self.y0 = int(y0)
        self.x1 = int(x1)
        self.y1 = int(y1)
