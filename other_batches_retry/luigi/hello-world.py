import luigi
import os
from luigi import Task, LocalTarget, WrapperTask
from luigi.parameter import DictParameter, Parameter, ListParameter
# os.environ["LUIGI_CONFIG_PATH"] = "home/raghav/table_detection/table/address_parser_api/other_batches_retry/config/luigi.cfg"
# print(os.environ["LUIGI_CONFIG_PATH"])
import time
import logging
from pathlib import Path 
import sys
# # print(Path(__file__))
# Project_DIR = Path(__file__).resolve().parents[0]
# print(Project_DIR)
# sys.path.append(str(Project_DIR))
logger = logging.getLogger(__name__)

class HelloLuigi(luigi.Task):

    local_target_name =Parameter()
    # print(local_target_name)

    def output(self):

        return luigi.LocalTarget(os.path.join(local_target_name,'hello-luigi.txt'))

    def run(self):
        logger.info("printing hello world output",extra={"class-name": HelloLuigi.__name__, "func-name": self.run.__name__})
        with self.output().open("w") as outfile:
            outfile.write("Hello Luigi!")

def execute(dir):
    
    luigi.build([HelloLuigi(local_target_name=dir)], workers=5)
    time.sleep(10)
 



if __name__ == '__main__':
    # luigi.run(main_task_cls=CreateReport,local_scheduler=False)
    # luigi.build(local_target_name=local_target_name)
    # luigi.run(main_task_cls=HelloLuigi)
    # while True:4
    logging.config.fileConfig("../config/logging4.conf",disable_existing_loggers=False)
    
    logger.info("starting luigi pipeline", extra={"class-name":"extractionpipeline", "func-name":"execute"},)
    dirname = "/home/raghav/table_detection/table/address_parser_api/other_batches_retry/data"
    local_target_name = os.path.join(dirname,"target")

    logger.info(f"the path has been shared {local_target_name}")

    while True:
        logger.info(f"starting luigi pipeline while condition", extra={"class-name":"extractionpipeline", "func-name":"execute"},)
        execute(local_target_name)
        
        
    
 