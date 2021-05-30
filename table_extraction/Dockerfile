FROM python:3.8-slim-buster

# Update and install depedencies
RUN apt-get update && apt-get install -y \
    autoconf automake build-essential curl git libsnappy-dev libtool pkg-config

RUN apt-get -y install tesseract-ocr
RUN apt-get -y install tesseract-ocr-ben
RUN apt-get -y install poppler-utils 

##imagemagic installation
RUN apt-get update -qq && apt-get install -y \
  ghostscript \
  libgs-dev \
  imagemagick


RUN tesseract --version

