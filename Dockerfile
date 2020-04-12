FROM python:3.6

COPY . /nevsky
RUN pip install /nevsky
WORKDIR /app

ENTRYPOINT [ "bash" ]