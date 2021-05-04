FROM python:3.7-slim-stretch

RUN apt-get update --fix-missing && apt-get install --no-install-recommends gcc pkg-config git curl build-essential ffmpeg libsm6 libxext6 libpcre3 libpcre3-dev -y 
RUN useradd --gid root --home-dir /usr/local/descriptor --create-home descriptor  

WORKDIR /usr/local/descriptor

COPY . ./

RUN chmod -R g+rwx /usr/local/descriptor && pip install --upgrade pip && pip install -r requirements.txt 
ENV TERM="xterm-256color"
EXPOSE 8080 

CMD ["uwsgi", "--uid", "descriptor", "--ini", "app.ini"]
