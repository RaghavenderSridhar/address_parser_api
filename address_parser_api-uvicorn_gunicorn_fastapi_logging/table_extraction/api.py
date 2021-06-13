import numpy as np

import pandas as pd

from flask import Flask, request, send_from_directory

from werkzeug.datastructures import FileStorage

from flask_restful import Resource, Api, reqparse

from flask_restful import fields, marshal

import base64

from io import BytesIO

from wand.color import Color

from wand.image import Image as WImage

import os, uuid

from pathlib import Path

from PIL import Image
from datetime import datetime
import logging
import json, pickle, cv2
from loggin import Filehandler, Formatter

from detector import init_detector, inference_detector

from distutils.util import strtobool
import mmcv

import configparser
import ssl

logging.captureWarnings("ignore")

from utils import ocr, deskew

app = Flas(__name__)

api = API(app)

##load model 

config_file ='model_config.py'

checkpoint_file ='model.pth'

model = init_detector(config_file, checkpoint_file, device='cpu'
app.logger.info("deep learning model has been loaded successfully")

config =configparser.ConfigParser()
config.read("./config/app_config.ini")

#3SSL cert
SSL_CERT_FILE = 


class TablePArser(Resourse):

    upload_parser = reqparse.RequestParser(bundle_errors=True)
    upload_parser.add_argument("file", location ='files', type =FileStorage, required=True)

    def get(self):
        return{"message":" table Parser service is up an running"}

    def post(self):

        result = inference_detector(model, image_array)
        has_table = check_has_table(result)

        




