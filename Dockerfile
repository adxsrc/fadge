FROM	python:slim
RUN	apt update && apt install time
RUN	pip install https://github.com/adxsrc/xaj/archive/refs/heads/main.zip
RUN	pip install https://github.com/adxsrc/fadge/archive/refs/heads/main.zip
