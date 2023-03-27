FROM python:3.9.7
ADD requirements.txt /
RUN pip install -r requirements.txt
ADD ./dash_housing_app.py /
ADD ./ETL.py /
ADD ./Model.py /
EXPOSE 8060
CMD ["python", "./dash_housing_app.py"]