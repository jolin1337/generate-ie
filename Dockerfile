FROM alpine:3.7
MAINTAINER <johannes.linden@miun.se>
FROM python:3.7

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

WORKDIR /app
CMD python --version
