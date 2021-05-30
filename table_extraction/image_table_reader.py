import cv2
import json
import numpy as np
import pandas as pd
import os
import logging
import configparser
import argparse
import datetime
import time
import pytesseract
import copy

print("test")

class ImageTableBorderReader:

    def __init__(self, pytesseract_path, temp_folder, debug_mode):

        self.pytesseract_path=pytesseract_path
        self.temp_folder=temp_folder
        self.debug_mode = debug_mode
        pytesseract.pytesseract.tesseract_cmd = pytesseract_path
    
    def extract(self, img, img_name, save_folder_path =None):

    def detect_all_cells(self,img,img_name,process_2_reqd=False, process_3_reqd=False):
        try:

            if len(img.shape)==3:
                img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if process_2_reqd == True or process_3_reqd == True:
                img_bin = cv2.adaptiveThreshold()


