FROM python:3.13.4

WORKDIR /app

COPY . .


RUN pip3 install -r requirements.txt

USER myuser

CMD ["python", "train.py"]