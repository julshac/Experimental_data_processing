import matplotlib.gridspec as gr
from inout.plot import *
from inout.fopen import *
from analysis.image_analysis import *
import numpy as np
from matplotlib import pyplot as plt
from model import Fourier
from analysis import stats
from analysis import amplitude_modulation
from model import trend, random, shifts, _fourier
import model._fourier as m
from scipy import ndimage
from tqdm import tqdm
from numba import jit
import cv2
from dataclasses import dataclass
from typing import List
from two_dimensional import erosionDilation
from test import MSRCR

@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass
class Stone:
    points: List[Point]

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return (i for i in self.points)


def get_stones(matrix):
    point_index = {}
    stones = []
    current_stone_coords_list = []
    old = False
    for i in tqdm(range(len(matrix))):
        for j in range(len(matrix[i])):
            if len(current_stone_coords_list) > 0 and matrix[i][j] == 0:
                if not old:
                    stones.append(Stone(current_stone_coords_list))
                for stone_coord in current_stone_coords_list:
                    point_index[stone_coord] = current_stone_coords_list
                old = False
                current_stone_coords_list = []
            if Point(i - 1, j) in point_index:
                old = True
                for p in set(current_stone_coords_list) - set(point_index[Point(i - 1, j)]):
                    point_index[Point(i - 1, j)].append(p)
                current_stone_coords_list = point_index[Point(i - 1, j)]
            if matrix[i][j] == 255:
                current_stone_coords_list.append(Point(i, j))
    return stones


def stonesSecond(threshimg):
    counter = 0
    picture1 = threshimg.copy()
    center_points = np.zeros_like(threshimg)
    result = get_stones(threshimg)
    for i in result:
        max_height_value = max(i, key=lambda x: x.y).y - min(i, key=lambda x: x.y).y
        max_weight_value = max(i, key=lambda x: x.x).x - min(i, key=lambda x: x.x).x
        if max_height_value == 6 and max_weight_value == 6:
            for j in i:
                center_points[j.x, j.y] = 255
                picture1[j.x, j.y] = 20
            # x = int(np.mean([x.x for x in i]))
            # y = int(np.mean([x.y for x in i]))
            # center_points[x, y] = 255
            # picture1[x, y] = 0
            counter += 1
    print(f"Stones count is {counter}")

    img = np.stack([center_points, picture1, picture1], axis=-1)
    plt.imshow(img)
    plt.show()


def stonesFirst(threshimg):
    kernel = np.ones((5, 5), np.uint8)  # Морфологические образы
    plt.title('Оригинальное пороговое изображение')
    plt.imshow(threshimg, cmap='gray')
    plt.figure()
    erosion = cv2.erode(threshimg, kernel, iterations=1)
    plt.title('Эрозия')
    plt.imshow(erosion, cmap='gray')
    plt.show()
    counter = 0
    stones = get_stones(erosion)
    center_points = np.zeros_like(threshimg)
    picture1 = threshimg.copy()
    for i in stones:
        if len(i) == 1:
            for j in i:
                # center_points[j.x, j.y] = 255
                # picture1[j.x, j.y] = 20
                center_points[j.x - 2: j.x + 2, j.y - 2: j.y + 3] = np.full((4, 5), 255)
                picture1[j.x - 2: j.x + 2, j.y - 2: j.y + 3] = np.full((4, 5), 20)
            counter += 1
    print(f"\nStones count is {counter}")
    img = np.stack([center_points, picture1, picture1], axis=-1)
    plt.imshow(img)
    plt.show()


def main():
    picture = to_one_channel(img_values("data/stones.jpg"))
    threshimg = np.zeros_like(picture)
    w = picture > 140
    b = picture < 90
    threshimg[w], threshimg[b] = 255, 0

    # threshimg = MSRCR(threshimg)
    stonesFirst(threshimg)
    stonesSecond(threshimg)


if __name__ == "__main__":
    main()

# if __name__ == '__main__':
    # def nothing(*arg):
#     #     pass


# cv2.namedWindow( "result" )
# cv2.namedWindow( "settings" )
#
# cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
# cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
# cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
# cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('v2', 'settings', 255, 255, nothing)
# cv2.createTrackbar('bloor', 'settings', 0, 10, nothing)
#
# while True:
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
#
#     bloor = cv2.getTrackbarPos('bloor', 'settings')
#     h1 = cv2.getTrackbarPos('h1', 'settings')
#     s1 = cv2.getTrackbarPos('s1', 'settings')
#     v1 = cv2.getTrackbarPos('v1', 'settings')
#     h2 = cv2.getTrackbarPos('h2', 'settings')
#     s2 = cv2.getTrackbarPos('s2', 'settings')
#     v2 = cv2.getTrackbarPos('v2', 'settings')
#
#     if bloor % 2 == 0:
#         bloor = bloor + 1
#
#     h_min = np.array((h1, s1, v1), np.uint8)
#     h_max = np.array((h2, s2, v2), np.uint8)
#
#
#     hsv = cv2.GaussianBlur(hsv, (bloor, bloor), 2)
#     thresh = cv2.inRange(hsv, h_min, h_max)
#
#     cv2.imshow('result', thresh)
#
#     ch = cv2.waitKey(5)
#     if ch == 27:
#         break