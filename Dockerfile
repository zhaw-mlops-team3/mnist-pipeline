FROM python:3.12

WORKDIR /app

COPY main.py evaluate.py train.py model.py requirements.txt sweep.yaml run_sweep.py ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "run_sweep.py"]
