from pytesseract import pytesseract
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import json
import cv2
from pytesseract import Output
pytesseract.tesseract_cmd ="usr/bin/tesseract"

def deskew(im,max_skew=10):

    if not isinstance(im,np.ndarray):

        raise ValueError("input error")

    height, width = im.shape[:2]

    im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastN1MeansDenoising(im_gs, h=3)
    im_bw = cv2.threshold(im_gs, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    lines = cv2.HoughLinesP(im_bw, 1, np.pi/360, 200, minLineLength = width /12, maxLineGap =width /150)


    angles = []
    for line in lines:

        x1,y1,x2,y2 =line[0]
        angles.append(np.arctan2(y2-y1,x2-x1))

    landscape = np.sum([abs(angle) > np.pi /4 for angle in angles]) > len(angles)/2

    if landscape:
        angles = [
            angle for angle in angles 
            if np.deg2rag(90-max_skew) < abs(angle) < np.deg2rag(90+max_skew)]
    else:
        angles =[angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        return im

    angle_deg = np.rad2deg(np.median(angles))

    deg = angle_deg

    if landscape:

        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg -= 90
    M = cv2.getRotationMatrix2D((width/2, height/2), angle_deg,1)

    if landscape:
        im = cv2.warpAffine(im, M (height, width), borderMode=cv2.BORDER_REPLICATE)
    else:
        im = cv2.warpAffine(im, M (height, width), borderMode=cv2.BORDER_REPLICATE)

    return im, deg

def check_has_table(result):

    if not isinstance(result, (tuple, list)):
        raise ValueError("some problem in table detection, result is not valid")

    if len(result)> 0:
        if len(result[0][0])>0 or len(result[0][2]) > 0:
            return True
    return False



    