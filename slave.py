import cv2 as cv
import numpy as np


class Bottle:

    def __init__(self, conf_threshold=0.5):
        self.confThreshold = conf_threshold

    def Objname(self):
        ObjectNames = []
        ObjectFile = 'coco.names'
        with open(ObjectFile, 'r') as ObjNam:
            ObjectNames = ObjNam.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weighPath = 'frozen_inference_graph.pb'

        self.net = cv.dnn_DetectionModel(weighPath, configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        return ObjectNames

    def reorder(self, MyPoints):
        MyPoints = MyPoints.reshape((4, 2))
        MyPointsNew = np.zeros((4, 1, 2), np.int32)
        add = MyPoints.sum(0)

        MyPointsNew[0] = MyPoints[np.argmin(add)]
        MyPointsNew[3] = MyPoints[np.argmax(add)]
        diff = np.diff(MyPoints, axis=1)
        MyPointsNew[1] = MyPoints[np.argmin(diff)]
        MyPointsNew[2] = MyPoints[np.argmax(diff)]
        return MyPointsNew

    def WarpImage(self, image, x, y, w, h, Width, Height):
        pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        pts2 = np.float32([[0, 0], [Width, 0], [0, Height], [Width, Height]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        imgOut = cv.warpPerspective(image, matrix, (Width, Height))

        imgCropped = imgOut[0:imgOut.shape[0], 0:imgOut.shape[1]]  # pixel change reduced to 20.
        imgCropped = cv.resize(imgCropped, (Width, Height))
        return imgCropped

    def ObjectDec(self, image, ObjectNames):
        ObjectIds, confid, bbox = self.net.detect(image, self.confThreshold)
        boundary = []
        if len(ObjectIds) != 0:
            for ids, value in enumerate(ObjectIds):
                if value == 44:
                    for ObjectIds, confident, box in zip(ObjectIds.flatten(), confid.flatten(), bbox):
                        boundary = box
                        cv.rectangle(image, box, color=(255, 255, 255), thickness=3)
                        cv.putText(image, ObjectNames[ObjectIds - 1], (box[0] + 10, box[1] - 50),
                                   cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)

        return boundary
