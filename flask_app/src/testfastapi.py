import uvicorn
from fastapi import FastAPI, Request
import urllib
import uuid
import os
import re
from postal.parser import parse_address
from postal.expand import expand_address


app = FastAPI()


@app.get('/')
def hello():
    return 'Hello, World!'
 
@app.post('/api/v1/addressparser')
async def addressparser(address : str):
    '''
    addressparser prediction putput
    '''
    try:
        result=parse_address(address)
        response = {'result':result}
        return {"message":response}
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8686, debug=True)


# @app.get("/")
# def home(name: str):
#     return {"message": f"Hello! {name}"}
# if __name__ == "__main__":
#     uvicorn.run(app, host='127.0.0.1', port=8686, debug=True)