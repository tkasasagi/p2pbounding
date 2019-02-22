import random
from os import listdir, remove, mkdir, path
from functions import *
import lycon
from skimage import io
import mlcrate as mlc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import shutil
from PIL import Image, ImageDraw, ImageFont


def choosefile(lis, number, seed = 777):
    random.seed(seed)
    return random.sample(lis, number)

def checkdir(directory):
    if not (path.isdir(directory)):
        mkdir(directory)
        
def image_resize(file, size):
    im = io.imread(file)
    imresized = resize(im, (size, size))
    return imresized

def f1score(tp, fp, fn):
    result = (2*tp) / ((2*tp) + fp + fn)
    return round(result, 4)

def getchar(df,charlist):
    all_predict = pd.DataFrame()
    for char in charlist:
        char_df = pd.DataFrame()        
        char_df = df.loc[df['char'] == char]
        
        all_predict = pd.concat([all_predict, char_df])
        
    all_predict.to_csv("./getchar_u/" + file, index = False)

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

def drawbox(im, coordinate): 
    d = ImageDraw.Draw(im)
    thickness = 5
    color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    x = coordinate['x'].tolist()
    y = coordinate['y'].tolist()
    w = coordinate['w'].tolist()
    h = coordinate['h'].tolist()
    for j in range(len(x)):    
        top_left = (x[j], y[j])
        bottom_right = (x[j]+w[j], y[j]+h[j])
        outline_color = color[j%len(color)]
        draw_rectangle(d, (top_left, bottom_right), color=outline_color, width=thickness)
    return d

def getiou(bb1, bb2):
    assert bb1[0] < bb1[1]
    assert bb1[2] < bb1[3]
    assert bb2[0] < bb2[1]
    assert bb2[2] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou