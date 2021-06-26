FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y \
    autoconf automake build-essential curl git libsnappy-dev libtool pkg-config

RUN echo " this is docker container for aother batches"

RUN echo " this is Maintained by Raghavender SRIDHAR"

LABEL maintainer="Raghavender Sridhar"

USER root

ARG WB_USER_HOME = /usr/workbench

ENV WB_HOME = ${WB_USER_HOME}

ENV LUIGI_CONFIG_PATH = /usr/workbench/config/luigi.config

RUN mkdir ${WB_USER_HOME}

ADD /entrypoint.sh /entrpoint.sh

WORKDIR ${WB_USER_HOME}

ADD . $(WB_USER_HOME)

USER root

RUN chmod +x ./entrpoint.sh \
    && pip install -r requirements.txt

EXPOSE 8082

ENTRYPOINT ["./entrypoint.sh"]



