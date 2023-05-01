FROM python:3.9
ADD requirements.txt /
RUN pip install -r requirements.txt
Add /. /
CMD ["python", "./dash_housing_app.py"]