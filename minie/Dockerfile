FROM maven:3.6.3-adoptopenjdk-8
WORKDIR /opt
RUN apt-get install git
RUN git clone https://github.com/uma-pi1/minie
WORKDIR /opt/minie
CMD ["mvn", "clean", "compile", "exec:java"]
