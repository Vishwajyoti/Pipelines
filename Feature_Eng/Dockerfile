FROM python:3.7.3-slim

WORKDIR /home/vishwajyoti_pandey/component/Feature_Eng/

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python","src/feature_eng.py"]
