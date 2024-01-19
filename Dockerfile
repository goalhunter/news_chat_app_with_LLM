FROM python:3.10.6

WORKDIR /app

COPY . /app

Run pip install -r requirements.txt

CMD ["streamlit","run","app.py"]