FROM python:3.12
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y 
# udating all the packages before deployment^^
RUN pip install -r requirements.txt
CMD ["python3","app.py"]
