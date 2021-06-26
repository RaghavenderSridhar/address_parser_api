import sys
from pathlib import Path
import configparser
import argparse
import json
import datetime
import logging
PROJECT_DIR =Path(__file__).parents[0]
print(PROJECT_DIR)
sys.path.append(str(PROJECT_DIR))
import os
logging.config.fileConfig("./logging.conf",disable_existing_loggers=False)
import uuid
import requests

logger = logging.getLogger(__name__)

class Housekeeping:

    def __init__(self,config):
        self.config=config

    def test(self):

        print("Hellow World")


if __name__ =="__main__":
    config =configparser.ConfigParser()
    config.optionxform =str
    config.read("config.ini",encoding="cp1251")
    housekeeping =Housekeeping(config)
    try:
        housekeeping.test()
    except Exception as e:
        logger.exception(f"Uh-Oh!, {e.__class__} occured",extra =("class-name": Housekeeping.__name__),)