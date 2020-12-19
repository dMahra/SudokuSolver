import cv2 as cv
from skimage.segmentation import clear_border
import numpy as np
from PIL import Image
from matplotlib import cm
from tensorflow.keras.models import load_model
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt


def initializeModel():
    model = load_model('numberModel.h5', compile=False)
    return model

def remove_int(digit, x, y):
    idx = (y != digit).nonzero()
    return x[idx], y[idx]

##preprocessing image
def preProcess(img):
    imgFinal = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # grayscale image
    imgFinal = cv.GaussianBlur(imgFinal, (5, 5), 1)  # Blur image
    imgFinal = cv.adaptiveThreshold(imgFinal, 255, 1, 1, 11, 2)  # apply adaptive threshold
    return imgFinal


##finding the largest contour which is the Sudoku Puzzle borders out of all the contours(our argument)
def largestContour(contours):
    largest_points = np.array([])
    max_area = 0
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 50:
            perimeter = cv.arcLength(contour, True)
            corner_points = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > max_area and len(corner_points) == 4:
                max_area = area
                largest_points = corner_points
    return largest_points, max_area


##reorder points correctly to use the warp perspective function
def reorder(ourPoints):
    ourPoints = ourPoints.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)
    add = ourPoints.sum(1)
    newPoints[0] = ourPoints[np.argmin(add)]
    newPoints[3] = ourPoints[np.argmax(add)]
    difference = np.diff(ourPoints, axis=1)
    newPoints[1] = ourPoints[np.argmin(difference)]
    newPoints[2] = ourPoints[np.argmax(difference)]
    return newPoints


##split puzzle into 81 different images
def splitboxes(image):
    rows = np.vsplit(image, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for col in cols:
            boxes.append(col)
    return boxes


## get prediction for all images
def cropp(npy_pic, crop_width, crop_height):
    # pil_img = Image.fromarray(np.uint8(npy_pic)).convert('RGB')
    pil_img = Image.fromarray(np.uint8(cm.gist_earth(npy_pic)*255))
    # height, width = image.shape[:2]
    # startrow, startcol = int(height * 0.2), int(width * 0.2)
    # endrow, endcol = int(height * 0.8), int(width * 0.8)
    # cropped_image = image[startrow:endrow, startcol:endcol]
    # return cropped_image
    img_width, img_height = pil_img.size
    c = pil_img.crop(((img_width - crop_width) // 2, (img_height - crop_height) // 2, (img_width + crop_width) // 2, (img_height + crop_height) // 2 + 2))
    # c = pil_img.crop((10, 10, 40, 44))
    c = cv.resize(np.array(c), (28, 28))
    c = np.array(c)[:, :, 0]  # 28, 28
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # c = cv.filter2D(c, -1, kernel)
    c = np.array([c])
    return c # 1, 28, 28 image

def check_center_crop(npy_pic):
    # pil_img = Image.fromarray(np.uint8(npy_pic)).convert('RGB')
    pil_img = Image.fromarray(np.uint8(cm.gist_earth(npy_pic) * 255))
    # pil_img = pil_img.convert('1')
    # img_width, img_height = pil_img.size
    c = pil_img.crop((14, 30, 32, 44))
    # c = pil_img.crop((10, 20, 20, 10))
    c = c.convert('1')
    c = np.array(c)
    # c = cv.resize(np.array(c), (28, 28))
    # c = cv.resize(np.array(c), (28, 28))
    # c = np.array(c)[:, :, 0]  # 28, 28
    # c = np.array([c]) # 1, 28, 28
    return c

    # c = cv.resize(np.array(c), (28, 28))
    # c = np.array(c)[:, :, 0]  # 28, 28
    # c = np.array([c]) # 1, 28, 28
    # return num_white_pix