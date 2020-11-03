FROM jolin1337/base-environement:v1.0


USER root
# Install java
RUN echo "deb http://security.debian.org/debian-security stretch/updates main" >> /etc/apt/sources.list && apt-get update
RUN apt-get install -y openjdk-8-jdk openjdk-8-jre

USER dev


COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

WORKDIR /home/dev/workspace/stanford-corenlp
CMD python --version
