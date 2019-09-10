FROM python:3.7.3-slim

WORKDIR /home/vishwajyoti_pandey/component/Training/

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python","src/training.py"]
