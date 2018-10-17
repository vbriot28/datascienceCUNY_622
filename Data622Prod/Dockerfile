FROM ubuntu:16.04

MAINTAINER Valerie Briot vcbriot@gmail.com

# Python & Pip
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Install Package
RUN pip3 install 	numpy \
                	scipy \
                	pandas \
	                scikit-learn

# Install awscli
RUN pip3 install awscli --upgrade --user

WORKDIR usr/local/bin

COPY ./src/pull_data.py .
COPY ./src/train_model.py .
COPY ./src/score_model.py .
COPY ./src/scripts_ini.sh .

#ENTRYPOINT ["/bin/sh", "./scripts_ini.sh"]
#ENTRYPOINT ["python3"]
