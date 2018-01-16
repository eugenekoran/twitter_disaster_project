FROM gw000/keras:2.0.8-py2-th-cpu

RUN pip --no-cache-dir install flask-restful

WORKDIR /srv/ai

EXPOSE 5000

COPY . .

CMD ["python", "api.py"]
