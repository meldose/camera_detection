import cv2 as cv
import numpy as np
import Slave as Sl

WidthImg = 640
HeightImg = 480

vid = cv.VideoCapture(1)
vid.set(3, WidthImg)
vid.set(4, HeightImg)

detector = Sl.Bottle(conf_threshold=0.75)
Object_names = detector.Objname()



def algorithm(image, b_w, b_y):
    height = image.shape[0]
    width = image.shape[1]
    
    # find the level of the water present:

    Level = height - b_y
    return Level



def ml_conversion(water_ml):
    water_ml = int((water_ml / 3))

    if water_ml < 101:
        print("Low Level of fluid is present =", water_ml, "ml")
    elif 100 < water_ml < 200:
        print("Medium Level of fluid is present =", water_ml, "ml")
    elif water_ml > 199:
        print("High Level of fluid is present =", water_ml, "ml")


def level_detection(image):
    grey_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("GRAY", grey_img)
    blur_image = cv.GaussianBlur(image, (7, 7), 0)
    canny = cv.Canny(blur_image, 190, 250)
    cv.imshow("CANNY", canny)

    lines = cv.HoughLines(canny, 1, np.pi / 180, 30, np.array([]))
   
    x = 0
    y = 0

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

       
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        if 0 < angle < 1:
            # print(angle)
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
           
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2

    return x, y


def process(org_image):
    image = cv.split(org_image)[0]

    _, thresh = cv.threshold(image, 150, 255, cv.THRESH_BINARY_INV)
    cv.imshow("Bottle", thresh)

    contour, hierarchy = cv.findContours(thresh, 1, 2)

    cx = 0
    cy = 0
    for cnt in contour:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        epsilon = 0.1 * perimeter

        approx = cv.approxPolyDP(cnt, epsilon, True)
        if 100 < area < 5000:
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.drawContours(org_image, [approx], -1, (0, 255, 255), 5)
    return cx, cy


def main(image):
    x, y = level_detection(image=image)
    print("x, y :", x, y)
    if y > 0:
        water_ml = algorithm(image=image, b_w=x, b_y=y)
        print("Level in pixel value", water_ml)
        ml_conversion(water_ml)
    else:
        print("Level of Water is not visible to detect")


while True:
    done, image = vid.read()
    boundary = detector.ObjectDec(image, ObjectNames=Object_names)
    if len(boundary) == 4:
        print(boundary)
        x, y, w, h = boundary[0], boundary[1], boundary[2], boundary[3]
        Width = w
        Height = h
        cropped_one = detector.WarpImage(image, x, y, w, h, Width, Height)
        if len(boundary) > 0:
            cv.imshow("CropImage", cropped_one)
            main(cropped_one)
        else:
            break
    else:
        print("no boundary no bottle detected")

    cv.imshow("Bottle", image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
