FROM python:3.12

WORKDIR /app

COPY train.py model.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]
