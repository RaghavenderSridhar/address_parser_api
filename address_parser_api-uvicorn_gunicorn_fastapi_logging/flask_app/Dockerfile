FROM python:3.8-slim-buster



RUN apt-get update && apt-get install -y \
    autoconf automake build-essential curl git libsnappy-dev libtool pkg-config

RUN echo " this is docker container for address parser"

RUN echo " this is Maintained by Raghavender SRIDHAR"

LABEL maintainer="Raghavender Sridhar"

USER root

###### installing pip dependency
COPY ./pip.conf /home/pip.conf

ENV PIP_CONFIG_FILE /home/pip.conf

ARG COMMIT
ENV COMMIT ${COMMIT:-master}
ENV DEBIAN_FRONTEND noninteractive


RUN git clone https://github.com/openvenues/libpostal -b $COMMIT

COPY ./src/build_libpostal.sh /libpostal/

WORKDIR /libpostal

RUN chmod +x *.sh

RUN ./build_libpostal.sh

# ####pip install

COPY requirements.txt /tmp/requirements.txt


RUN python3 -m pip install -r /tmp/requirements.txt


EXPOSE 8080

# CMD bash

# ###entrypoint

WORKDIR /addressParser

ADD ./src /addressParser/flask_app/

ADD ./src/entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]