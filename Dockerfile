FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04 

COPY requirements.txt requirements.txt

RUN pip install --timeout 3600 -r requirements.txt

COPY . .

CMD python3 jawi_api.py

EXPOSE 8000
