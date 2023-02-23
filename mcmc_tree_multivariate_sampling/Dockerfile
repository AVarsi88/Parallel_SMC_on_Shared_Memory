FROM python:3.8

ADD Metrics.py .
ADD create_tree.py .
ADD example_multi_core.py .
ADD Test_tree.py .
ADD Tree_sample.py .




RUN pip install -U scikit-learn pandas numpy multiprocess

CMD ["python", "./example_multi_core.py"]