FROM jolin1337/base-environment:v1.0

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

WORKDIR /home/dev/workspace/stanford-corenlp
CMD python --version
