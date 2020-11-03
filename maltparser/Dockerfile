FROM maven:3.6.3-adoptopenjdk-8

WORKDIR /opt/maltparser
ADD http://maltparser.org/dist/maltparser-1.7.2.tar.gz maltparser-1.7.2.tar.gz
ADD http://www.maltparser.org/mco/swedish_parser/swemalt-1.7.2.mco swemalt-1.7.2.mco
RUN tar -zxvf maltparser-1.7.2.tar.gz
RUN mv maltparser-1.7.2/maltparser-1.7.2.jar maltparser-1.7.2.jar
RUN mv maltparser-1.7.2/lib maltparserlib

ADD http://mumin.ling.su.se/projects/stagger/snapshot.tar.bz2 stagger.tar.bz2
ADD http://mumin.ling.su.se/projects/stagger/swedish.bin.bz2 swedish.bin.bz2
RUN bunzip2 swedish.bin.bz2
RUN bunzip2 stagger.tar.bz2 && tar -xf stagger.tar
RUN mv stagger/stagger.jar stagger.jar
RUN rm -rf stagger.tar stagger maltparser-1.7.2  maltparser-1.7.2.tar.gz
#RUN cd icestagger/src && make
