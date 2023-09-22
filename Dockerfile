FROM python:3.9
RUN pip install pandas xgboost scikit-learn numpy 
COPY . .
CMD ["python", "web.py"]
EXPOSE 8080

