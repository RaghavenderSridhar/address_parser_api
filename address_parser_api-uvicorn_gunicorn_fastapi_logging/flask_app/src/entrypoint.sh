#!/bin/sh
set -e

# python /addressParser/flask_app/app.py
cd /addressParser/flask_app/

# gunicorn -b localhost:8080 -w 4 wsgi:app
# gunicorn -w 3 -b :5000 -t 30 --reload wsgi:app
# gunicorn -b :5000 -k uvicorn.workers.UvicornWorker testfastapi:app 
gunicorn -b :5000 --log-config logging.conf -k uvicorn.workers.UvicornWorker testfastapi:app


echo "raghav is great"

exec "$e";