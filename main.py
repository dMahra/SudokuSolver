import cv2

print('setting up...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2 as cv
from utils import *


# pathImage = '/Users/darsh_mahra/PycharmProjects/numberrecognition/media/secondhand.jpg'
pathImage = '/Users/darsh_mahra/Documents/GitHub/SudokuSolver/numberrecognition/media/secondhand.jpg'


heightimage, widthimage = 450, 450
Model = initializeModel()

##preparing the image
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthimage, heightimage))  # make image a square
imgFinal = preProcess(img)

##finding all the contours
imgContours = img.copy()
imgBigContour = img.copy()
dk, contours, hierarchy = cv2.findContours(imgFinal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0),3)  # draw the contours we just found over original image copy(imgcontours)

# -----RUNTHIS-----
# cv2.imshow('myimage', imgContours)
# cv2.waitKey(0)

##find the largest contour
largest, maxArea = largestContour(contours)
# print(largest) # ==> four points of the largest contour(not ordered)
largest = reorder(largest)
# print(largest)
if largest.size != 0:
    point1, point2 = np.float32(largest), np.float32([[0, 0], [widthimage, 0], [0, heightimage], [widthimage, heightimage]])
    matrix = cv2.getPerspectiveTransform(point1, point2)
    imgWarped = cv2.warpPerspective(img, matrix, (widthimage, heightimage))
    # imgWarped = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
    # cv2.drawContours(imgBigContour, largest, -1, (0, 255, 0), 10)


# cv.imshow('warped', imgWarped)
# cv.waitKey(0)

## extracting digits and creating puzzle's grid

# here grid is the cropped image
grid = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY) # VERY IMPORTANT
# Adaptive thresholding the cropped grid and inverting it
grid = cv.bitwise_not(cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 1))

edge_h = np.shape(grid)[0]
edge_w = np.shape(grid)[1]
celledge_h = edge_h // 9
celledge_w = np.shape(grid)[1] // 9

tempgrid = []
for i in range(celledge_h, edge_h + 1, celledge_h):
    for j in range(celledge_w, edge_w + 1, celledge_w):
        rows = grid[i - celledge_h:i]
        tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

# Creating the 9X9 grid of images
finalgrid = []
for i in range(0, len(tempgrid) - 8, 9):
    finalgrid.append(tempgrid[i:i + 9])
# Converting all the cell images to np.array
for i in range(9):
    for j in range(9):
        finalgrid[i][j] = np.array(finalgrid[i][j])
try:
    for i in range(9):
        for j in range(9):
            os.remove("BoardCells/cell" + str(i) + str(j) + ".jpg")
except:
    pass
for i in range(9):
    for j in range(9):
        cv2.imwrite(str("BoardCells/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])
#
# grape = finalgrid[3][4][0]
# grape = cv.resize(grape, (28, 28))
# grape = np.array([grape])
# grape = custom_crop(grape)
# pred = Model.predict(grape)
# print('predicted integer: '+ str(np.argmax(pred)))
# darsh = check_center_crop(grape)
# aadarsh = cropp(grape, 32, 32)
# cv.imshow('warped', darsh[0]) # 5
# cv.waitKey(0)
# cv.imshow('warped', aadarsh[0]) # 5
# cv.waitKey(0)
# cv.imshow('warped', cropp(finalgrid[1][6], 32, 32))
# cv.waitKey(0)
# print(np.sum(check_center_crop(finalgrid[0][0]) == 255)) #white pixels?
# print(check_center_crop(finalgrid[0][3]))
# print(np.sum(check_center_crop(finalgrid[1][6])) == 255)
print(np.sum(check_center_crop(finalgrid[1][6]) == 0) > 250)
# print(np.sum(check_center_crop(finalgrid[0][0]) == 0))

# cv.imshow('example', check_center_crop(finalgrid[0][0]))
# cv.waitKey(0)
## main
for i in range(9):
    for j in range(9):
        pic = finalgrid[i][j] #picture before it's cropped
        # pic = cv.resize(pic, (28, 28))
        # pic = np.array([pic])
        # probValue = np.amax(Model.predict(cropp(pic, 32, 32)), axis=-1)
        # print(probValue)
        # print(probValue)
        if np.sum(check_center_crop(pic) == 0) > 245: #checks if cropped picture is black (out of 252 pixels)
            finalgrid[i][j] = 0
        else:
            finalgrid[i][j] = np.argmax(Model.predict(cropp(pic, 32, 32)))
        # if check_center_crop(finalgrid[i][j]) == 0:
        #     finalgrid[i][j] = 0
        # finalgrid[i][j] = cropp(pic)


print(np.array(finalgrid))
# print(np.array(finalgrid))

# pred = Model.predict(finalgrid[5][4])
# print('predicted integer: '+ str(np.argmax(pred)))
#



