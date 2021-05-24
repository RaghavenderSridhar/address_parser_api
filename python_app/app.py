import urllib
import uuid
import os
import re
from flask import Flask, request, jsonify
from postal.parser import parse_address
from postal.expand import expand_address
from flasgger import Swagger

app =Flask(__name__)
swagger = Swagger(app)

@app.route('/')
def hello():
    return 'Hello, World!'
 
@app.route('/api/v1/addressparser', methods=['POST'])
def addressparser():
    """Example endpoint returning details of address parser
    ---
    parameters:
      - name: address
        in: query
        type: string
        required: true
      
    responses:
      200:
        description: the address parser returns all the sub details 
    """
    try:
        # text = request.form['address']
        address_value = str(request.args.get("address"))
        # text = str(text)
        result=parse_address(address_value)
        response = {'result':result}
        return jsonify(response)
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port ="8080")

