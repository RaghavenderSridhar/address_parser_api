import urllib
import uuid
import os
import re
from flask import Flask, request, jsonify
from postal.parser import parse_address
from postal.expand import expand_address

app =Flask(__name__)
# api =API(app)


# class Address:

#     def __init__(self):
#         self.result = []
#         self.final_parser_address = {}

#     def parserv3(self, address):
#         address = re.sub(r"\([^)]*\)","", address)
#         result = parse_address(address)
#         return result

# class Parserv3(Resource):
#     def get(self, address):
#         add_parser = urllib.parse.unquote(address)
#         result = add_parser.parserv3(address)
#         return jsonify(result)

# api.add_resource(Parserv3,"/addressParserv3/<address>")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0",port ="8080")

# app = Flask(__name__)
 
# classifier = TextClassifier.load('en-sentiment')
@app.route('/')
def hello():
    return 'Hello, World!'
 
@app.route('/api/v1/addressparser', methods=['POST'])
def addressparser():
    text = request.form['address']
    text = str(text)
    result=parse_address(text)
    response = {'result':result}
    return jsonify(response), 200
 
if __name__ == "__main__":
    app.run(host="0.0.0.0",port ="8080")

