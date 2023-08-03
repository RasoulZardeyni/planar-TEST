import math
from scipy.spatial.distance import cdist
import numpy as np
import cv2
from termcolor import colored
import os
import matplotlib.pyplot as plt


def threshold(images, thresh, maxval, type):
    thresh_images = []
    for img in images:
        thresh_images.append(cv2.threshold(img, thresh, maxval, type)[1])
    return thresh_images


def get_corner(img, index):
    plot = False
    match index:
        case 0:
            # img = cv2.flip(img, 0)
            a = np.where(img == 0)[0][0]
            b = np.where(img.T == 0)[0][0]
            zero_coords = np.argwhere(img == 0)
            distances = cdist([(a, b)], zero_coords)
            nearest_zero_coord = zero_coords[distances.argmin()]
            if plot:
                image = cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (nearest_zero_coord[1], nearest_zero_coord[0]), 64, (255, 0, 0), 16)
                plt.imshow(image)
                plt.pause(2)
                plt.close()
            return (nearest_zero_coord[1], img.shape[0] - nearest_zero_coord[0])

        case 1:
            # img = cv2.flip(img, 0)
            a = np.where(img == 0)[0][0]
            b = np.where(img.T == 0)[0][-1]
            zero_coords = np.argwhere(img == 0)
            distances = cdist([(a, b)], zero_coords)
            nearest_zero_coord = zero_coords[distances.argmin()]
            if plot:
                image = cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (nearest_zero_coord[1], nearest_zero_coord[0]), 64, (255, 0, 0), 16)
                plt.imshow(image)
                plt.pause(2)
                plt.close()
            return (nearest_zero_coord[1], img.shape[0] - nearest_zero_coord[0])
        case 2:
            # img = cv2.flip(img, 1)
            a = np.where(img == 0)[0][-1]
            b = np.where(img.T == 0)[0][0]
            zero_coords = np.argwhere(img == 0)
            distances = cdist([(a, b)], zero_coords)
            nearest_zero_coord = zero_coords[distances.argmin()]
            if plot:
                image = cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (nearest_zero_coord[1], nearest_zero_coord[0]), 64, (255, 0, 0), 16)
                plt.imshow(image)
                plt.pause(2)
                plt.close()
            return (nearest_zero_coord[1], img.shape[0] - nearest_zero_coord[0])
        case 3:
            # img = cv2.flip(img, 1)
            a = np.where(img == 0)[0][-1]
            b = np.where(img.T == 0)[0][-1]
            zero_coords = np.argwhere(img == 0)
            distances = cdist([(a, b)], zero_coords)
            nearest_zero_coord = zero_coords[distances.argmin()]
            if plot:
                image = cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (nearest_zero_coord[1], nearest_zero_coord[0]), 64, (255, 0, 0), 16)
                plt.imshow(image)
                plt.pause(2)
                plt.close()
            return (nearest_zero_coord[1], img.shape[0] - nearest_zero_coord[0])
    


def get_positions(images_tile):
    corner_positions = []
    for img in images_tile:
        corner = get_corner(img)
        corner_positions.append(corner)
    return corner_positions

# R_REF = 16 * 6000
# C_REF = 4 * 12000

R_REF = 1200
C_REF = 600


def shift(sample_positions, images_tile):
    # for img in images_tile:
    #     plt.imshow(img)
    #     plt.pause(2)
    #     plt.close()
    # print(sample_positions)
    tile_positions = []
    for i in range(4):
        img = images_tile[i]
        tile_positions.append(get_corner(img, i))
        # print(colored(tile_positions[-1], 'red'))
    # tile_positions[0] = (-1 * tile_positions[0][0], -1 * tile_positions[0][1])
    # tile_positions[1] = (-1 * tile_positions[1][0], tile_positions[1][1])
    # tile_positions[2] = (tile_positions[2][0], -1 * tile_positions[2][1])
    # tile_positions[3] = (tile_positions[3][0], tile_positions[3][1])
    # print(tile_positions)
    full_sample_positions = [(0, R_REF),
                             (C_REF, R_REF),
                             (0, 0),
                             (C_REF, 0)]
    # print(colored(get_size(positions=full_sample_positions), 'blue'))
    new_positions = []
    #METHOD 1
    for sp, tp, fs in zip(sample_positions, tile_positions, full_sample_positions):
        new_positions.append((tp[0] - sp[0] + fs[0], tp[1] - sp[1] + fs[1]))
    # print(colored(new_positions, 'green'))
    # METHOD 2
    # translation = [(sample_positions[i][0] - full_sample_positions[i][0], 
    #                sample_positions[i][1] - full_sample_positions[i][1])
    #                for i in range(4)]
    # for i in range(4):
    #     new_positions.append((
    #         tile_positions[i][0] + translation[i][0],
    #         tile_positions[i][1] + translation[i][1]
    #     ))
    
    
    return new_positions


def get_size(positions):
    width = (math.dist(positions[0], positions[1]) + math.dist(positions[2], positions[3])) / 2
    height = (math.dist(positions[0], positions[2]) + math.dist(positions[1], positions[3])) / 2
    diagonal_1 = math.dist(positions[0], positions[3])
    diagonal_2 = math.dist(positions[1], positions[2])
    diff_dig = abs(diagonal_1 - diagonal_2)
    # return width, height, diagonal_1, diagonal_2, diff_dig
    return round(width, 2), round(height, 2), round(diagonal_1, 2), round(diagonal_2, 2), round(diff_dig, 2)


sample_images = []
# folder_sample = 'images-amin\SAMPLE'
# folder_sample = 'images-amin\Sample_xd'
# folder_sample = 'images-amin\DR\sample'
folder_sample = 'images-amin\KH\sample'
# folder_sample = 'images-amin\MAKE LINE LIKE XD\SAMPLE'

tiles_images = []
sample_positions = []
# folder_tile = 'images-amin\sample1-d0.6mm'
# folder_tile = 'images-amin\sample2-d2mm'
# folder_tile = 'images-amin\sample3-d00mm'
# folder_tile = 'images-amin\Test_1'
# folder_tile = 'images-amin\sample3-d00mm-again'
# folder_tile = 'images-amin\Test'
# folder_tile = 'images-amin\Test_shift'
# folder_tile = 'images-amin\DR\Test_1'
folder_tile = 'images-amin/KH/700_1200'
# folder_tile = 'images-amin\MAKE LINE LIKE XD\sample1-d0.6mm'
xd_sample = True
if xd_sample:
    for path in os.listdir(folder_sample):
        if path[path.find('_') + 1] != '4' and path[path.find('_') + 1] != '5':
            p = os.path.join(folder_sample, path)
            sample_images.append(cv2.imread(p, 0))
            sample_images[-1] = cv2.medianBlur(sample_images[-1], 9)
            # sample_images[-1] = sample_images[-1][:, ::4]
            # if path[path.find('_') + 1] == '0' or path[path.find('_') + 1] == '2':
            #     sample_images[-1] = cv2.flip(sample_images[-1], 1)

    for path in os.listdir(folder_tile):
        if path[path.find('_') + 1] != '4' and path[path.find('_') + 1] != '5':
            p = os.path.join(folder_tile, path)
            tiles_images.append(cv2.imread(p, 0))
            tiles_images[-1] = cv2.medianBlur(tiles_images[-1], 9)
            # tiles_images[-1] = tiles_images[-1][:, ::4]
            # if path[path.find('_') + 1] == '0' or path[path.find('_') + 1] == '2':
            #     tiles_images[-1] = cv2.flip(tiles_images[-1], 1)
else:
    for path in os.listdir(folder_sample):
        match path[path.find('_') + 1]:
            case '0' | '1':
                p = os.path.join(folder_sample, path)
                sample_images.append(cv2.imread(p, 0))
                sample_images[-1] = cv2.flip(sample_images[-1], 0)
                sample_images[-1] = cv2.medianBlur(sample_images[-1], 9)
            case '2' | '3':
                p = os.path.join(folder_sample, path)
                sample_images.append(cv2.imread(p, 0))
                # sample_images[-1] = cv2.flip(sample_images[-1], 0)
                sample_images[-1] = cv2.medianBlur(sample_images[-1], 9)

    for path in os.listdir(folder_tile):
        match path[path.find('_') + 1]:
            case '0' | '1':
                p = os.path.join(folder_tile, path)
                tiles_images.append(cv2.imread(p, 0))
                tiles_images[-1] = cv2.flip(tiles_images[-1], 0)
                tiles_images[-1] = cv2.medianBlur(tiles_images[-1], 9)
            case '2' | '3':
                p = os.path.join(folder_tile, path)
                tiles_images.append(cv2.imread(p, 0))
                tiles_images[-1] = cv2.medianBlur(tiles_images[-1], 9)
BINARY_THRESH = 200
sample_images = threshold(sample_images, BINARY_THRESH, 255, cv2.THRESH_BINARY)
tiles_images = threshold(tiles_images, BINARY_THRESH, 255, cv2.THRESH_BINARY)
print(colored(folder_tile[folder_tile.rfind("-") + 2: ], 'red'))
for i in range(4):
    img = sample_images[i]
    sample_positions.append(get_corner(img, i))
    # print(colored(sample_positions[-1], 'red'))
# print(colored(get_size(positions=sample_positions), 'blue'))
for i in range(0, len(tiles_images), 4):
    positions = shift(sample_positions=sample_positions, images_tile=tiles_images[i:i + 4])
    size = get_size(positions=positions)
    print(colored(size, 'green'))
    # print(colored(np.abs(size[0] - 9900), 'yellow'))
# sample_positions[0] = (-1 * sample_positions[0][0], -1 * sample_positions[0][1])
# sample_positions[1] = (-1 * sample_positions[1][0], sample_positions[1][1])
# sample_positions[2] = (sample_positions[2][0], -1 * sample_positions[2][1])
# sample_positions[3] = (sample_positions[3][0], sample_positions[3][1])
# for i in range(0, len(tiles_images), 4):
#     positions = shift(sample_positions=sample_positions,
#                       images_tile=[tiles_images[i], tiles_images[i+1], tiles_images[i+2], tiles_images[i+3]])
#     print(get_size(positions))