FROM python:3.7.3-slim

WORKDIR /home/vishwajyoti_pandey/component/Testing/

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python","src/test.py"]
