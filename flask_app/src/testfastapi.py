import uvicorn
from fastapi import FastAPI, Request
import urllib
import uuid
import os
import re
from postal.parser import parse_address
from postal.expand import expand_address

from fastapi.logger import logger
# ... other imports
import logging
app = FastAPI()

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers




@app.get('/')
def hello():
    return 'Hello, World!'
 
@app.post('/api/v1/addressparser')
async def addressparser(address : str):
    '''
    addressparser prediction putput
    '''
    try:
        print(address)
        logger.info("successfully got the address")
        result=parse_address(address)
        if result:
            logger.info("successfully parsed the address")
        else:
            logger.info("failed to parsed the address")
        response = {'result':result}
        logger.info("successfully going to return the address")
        return {"message":response}
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    # uvicorn.run(app, host='127.0.0.1', port=8686, log_level="info")
    # uvicorn.run(app, host='127.0.0.1',reload=True, debug=True)
    logger.setLevel(gunicorn_logger.level)
    uvicorn.run(app, host='127.0.0.1')
else:
    logger.setLevel(logging.DEBUG)



# @app.get("/")
# def home(name: str):
#     return {"message": f"Hello! {name}"}
# if __name__ == "__main__":
#     uvicorn.run(app, host='127.0.0.1', port=8686, debug=True)