from continuumio/miniconda3:23.10.0-1

WORKDIR /app

RUN apt-get update && apt-get install build-essential -y
RUN apt-get update && apt-get install cmake -y
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl
RUN apt-get update && \
    apt-get install -y python3-dev unzip

COPY prepare.sh .
COPY requirements.txt .

RUN conda create -n moeva2 python=3.8.8 -y
RUN conda run -n moeva2 --no-capture-output pip install -r requirements.txt
RUN conda run -n moeva2 --no-capture-output pip install scipy==1.4.0

